


<a id='Training-a-Simple-LSTM'></a>

# Training a Simple LSTM


In this tutorial we will go over using a recurrent neural network to classify clockwise and anticlockwise spirals. By the end of this tutorial you will be able to:


1. Create custom Lux models.
2. Become familiar with the Lux recurrent neural network API.
3. Training using Optimisers.jl and Zygote.jl.


<a id='Package-Imports'></a>

## Package Imports


```julia
using Lux, LuxAMDGPU, LuxCUDA, JLD2, MLUtils, Optimisers, Zygote, Random, Statistics
```


<a id='Dataset'></a>

## Dataset


We will use MLUtils to generate 500 (noisy) clockwise and 500 (noisy) anticlockwise spirals. Using this data we will create a `MLUtils.DataLoader`. Our dataloader will give us sequences of size 2 × seq*len × batch*size and we need to predict a binary value whether the sequence is clockwise or anticlockwise.


```julia
function get_dataloaders(; dataset_size=1000, sequence_length=50)
    # Create the spirals
    data = [MLUtils.Datasets.make_spiral(sequence_length) for _ in 1:dataset_size]
    # Get the labels
    labels = vcat(repeat([0.0f0], dataset_size ÷ 2), repeat([1.0f0], dataset_size ÷ 2))
    clockwise_spirals = [reshape(d[1][:, 1:sequence_length], :, sequence_length, 1)
                         for d in data[1:(dataset_size ÷ 2)]]
    anticlockwise_spirals = [reshape(d[1][:, (sequence_length + 1):end], :,
        sequence_length, 1) for d in data[((dataset_size ÷ 2) + 1):end]]
    x_data = Float32.(cat(clockwise_spirals..., anticlockwise_spirals...; dims=3))
    # Split the dataset
    (x_train, y_train), (x_val, y_val) = splitobs((x_data, labels); at=0.8, shuffle=true)
    # Create DataLoaders
    return (
        # Use DataLoader to automatically minibatch and shuffle the data
        DataLoader(collect.((x_train, y_train)); batchsize=128, shuffle=true),
        # Don't shuffle the validation data
        DataLoader(collect.((x_val, y_val)); batchsize=128, shuffle=false))
end
```


```
get_dataloaders (generic function with 1 method)
```


<a id='Creating-a-Classifier'></a>

## Creating a Classifier


We will be extending the `Lux.AbstractExplicitContainerLayer` type for our custom model since it will contain a lstm block and a classifier head.


We pass the fieldnames `lstm_cell` and `classifier` to the type to ensure that the parameters and states are automatically populated and we don't have to define `Lux.initialparameters` and `Lux.initialstates`.


To understand more about container layers, please look at [Container Layer](http://lux.csail.mit.edu/stable/manual/interface/#container-layer).


```julia
struct SpiralClassifier{L, C} <:
       Lux.AbstractExplicitContainerLayer{(:lstm_cell, :classifier)}
    lstm_cell::L
    classifier::C
end
```


We won't define the model from scratch but rather use the [`Lux.LSTMCell`](../../api/Lux/layers#Lux.LSTMCell) and [`Lux.Dense`](../../api/Lux/layers#Lux.Dense).


```julia
function SpiralClassifier(in_dims, hidden_dims, out_dims)
    return SpiralClassifier(LSTMCell(in_dims => hidden_dims),
        Dense(hidden_dims => out_dims, sigmoid))
end
```


```
Main.var"##292".SpiralClassifier
```


We can use default Lux blocks – `Recurrence(LSTMCell(in_dims => hidden_dims)` – instead of defining the following. But let's still do it for the sake of it.


Now we need to define the behavior of the Classifier when it is invoked.


```julia
function (s::SpiralClassifier)(x::AbstractArray{T, 3},
    ps::NamedTuple,
    st::NamedTuple) where {T}
    # First we will have to run the sequence through the LSTM Cell
    # The first call to LSTM Cell will create the initial hidden state
    # See that the parameters and states are automatically populated into a field called
    # `lstm_cell` We use `eachslice` to get the elements in the sequence without copying,
    # and `Iterators.peel` to split out the first element for LSTM initialization.
    x_init, x_rest = Iterators.peel(eachslice(x; dims=2))
    (y, carry), st_lstm = s.lstm_cell(x_init, ps.lstm_cell, st.lstm_cell)
    # Now that we have the hidden state and memory in `carry` we will pass the input and
    # `carry` jointly
    for x in x_rest
        (y, carry), st_lstm = s.lstm_cell((x, carry), ps.lstm_cell, st_lstm)
    end
    # After running through the sequence we will pass the output through the classifier
    y, st_classifier = s.classifier(y, ps.classifier, st.classifier)
    # Finally remember to create the updated state
    st = merge(st, (classifier=st_classifier, lstm_cell=st_lstm))
    return vec(y), st
end
```


<a id='Defining-Accuracy,-Loss-and-Optimiser'></a>

## Defining Accuracy, Loss and Optimiser


Now let's define the binarycrossentropy loss. Typically it is recommended to use `logitbinarycrossentropy` since it is more numerically stable, but for the sake of simplicity we will use `binarycrossentropy`.


```julia
function xlogy(x, y)
    result = x * log(y)
    return ifelse(iszero(x), zero(result), result)
end

function binarycrossentropy(y_pred, y_true)
    y_pred = y_pred .+ eps(eltype(y_pred))
    return mean(@. -xlogy(y_true, y_pred) - xlogy(1 - y_true, 1 - y_pred))
end

function compute_loss(x, y, model, ps, st)
    y_pred, st = model(x, ps, st)
    return binarycrossentropy(y_pred, y), y_pred, st
end

matches(y_pred, y_true) = sum((y_pred .> 0.5) .== y_true)
accuracy(y_pred, y_true) = matches(y_pred, y_true) / length(y_pred)
```


```
accuracy (generic function with 1 method)
```


Finally lets create an optimiser given the model parameters.


```julia
function create_optimiser(ps)
    opt = Optimisers.ADAM(0.01f0)
    return Optimisers.setup(opt, ps)
end
```


```
create_optimiser (generic function with 1 method)
```


<a id='Training-the-Model'></a>

## Training the Model


```julia
function main()
    # Get the dataloaders
    (train_loader, val_loader) = get_dataloaders()

    # Create the model
    model = SpiralClassifier(2, 8, 1)
    rng = Random.default_rng()
    Random.seed!(rng, 0)
    ps, st = Lux.setup(rng, model)

    dev = gpu_device()
    ps = ps |> dev
    st = st |> dev

    # Create the optimiser
    opt_state = create_optimiser(ps)

    for epoch in 1:25
        # Train the model
        for (x, y) in train_loader
            x = x |> dev
            y = y |> dev
            (loss, y_pred, st), back = pullback(p -> compute_loss(x, y, model, p, st), ps)
            gs = back((one(loss), nothing, nothing))[1]
            opt_state, ps = Optimisers.update(opt_state, ps, gs)

            println("Epoch [$epoch]: Loss $loss")
        end

        # Validate the model
        st_ = Lux.testmode(st)
        for (x, y) in val_loader
            x = x |> dev
            y = y |> dev
            (loss, y_pred, st_) = compute_loss(x, y, model, ps, st_)
            acc = accuracy(y_pred, y)
            println("Validation: Loss $loss Accuracy $acc")
        end
    end

    return (ps, st) |> cpu_device()
end

ps_trained, st_trained = main()
```


```
((lstm_cell = (weight_i = Float32[-0.86109823 -0.41300547; -0.25317872 -0.75296676; 0.8150142 0.8886752; -0.063962206 0.07565517; -0.142488 -0.62122846; 0.28364563 0.5509813; -0.8397406 0.07917288; 0.08400166 0.040066984; -0.04076125 0.06566992; -0.6652266 0.45247388; 1.1580498 -0.0102329105; -0.015905563 0.088092476; -0.11248518 0.24690166; -0.8549774 0.3174361; -0.12737702 -0.26575273; 1.2182298 0.09449516; 0.6834873 -0.6284278; 0.12697554 -0.34102964; 0.44665834 -0.32216343; -0.54946107 0.51738673; 0.58378637 -0.8383898; 0.08600128 0.28561887; 0.8749789 -0.63989234; 0.9768136 0.6307094; -1.1746469 -0.11977771; 0.4777866 0.640712; 1.0544218 0.6956745; -0.4421481 -0.16938002; 0.74745625 -0.7041255; -0.3478823 0.7303035; -0.21784735 0.60330516; 0.62901944 -0.31009486], weight_h = Float32[-0.49901304 -0.055899635 0.33036724 -0.26838258 0.29215845 0.019177705 -0.6705474 0.54549026; -0.6676161 0.21137731 -0.05429091 0.6270167 0.370018 0.3095997 0.129925 -0.05085613; 0.011177453 0.062668316 -0.037292797 0.57014453 -0.73365736 0.31458724 -0.64560163 -0.16400333; 0.04162552 -0.26326796 0.8867419 -0.09656639 0.91057384 0.10385287 0.28637114 0.9057489; -0.35887212 0.46881345 0.7871058 0.30184513 0.40012684 0.64026976 -0.32747632 -0.07950502; -0.059890553 -0.33389518 0.32953566 -0.0923979 -0.29553974 -0.21587518 -0.48847258 0.028666953; -0.79343903 0.36852515 0.68173164 0.5315879 -0.8180451 0.6389902 0.01732006 -0.6370446; 0.6960166 0.3125069 1.0971912 -1.306187 0.7990777 0.1822408 -0.73783517 1.1304948; -0.16371365 0.47520044 0.43679968 -0.50483716 0.45330474 0.68901956 0.16107239 0.6304338; -0.42044428 -0.18461189 -0.38222447 -0.0052508228 0.23422799 -0.0023520393 -0.76537937 0.70049167; -0.15397923 0.5509462 -0.021535266 0.53180814 -0.5138755 0.2984505 -0.27314055 -0.38019106; 0.06183379 0.88042605 0.2878059 -0.113169774 0.82837147 0.649337 0.39616457 0.49716598; 0.6536262 0.4845957 0.3779326 -0.07135006 0.92149174 0.16737352 -0.63693285 0.44227043; -0.09729907 0.44274998 0.085673384 0.099373795 0.603692 -0.014884567 -0.15109728 -0.42437455; -0.8312972 0.33384126 0.08231428 -0.38942343 -0.26467472 0.799906 -0.3782752 0.40489414; -0.63534117 0.7859664 0.41068295 1.0495801 -0.09666797 0.81919605 -0.10524817 0.7641982; 0.14645678 -0.6436684 -0.06083165 -0.44403622 -0.35507008 -0.2380502 -0.16563074 0.17244928; -0.44310865 0.23241478 0.19428663 0.719458 0.3960184 -0.35381278 -0.36346537 0.6474627; -0.5578544 0.70807135 0.09320072 -0.42627987 -0.298255 0.74436253 0.04635552 0.60438645; -0.7780322 0.3406372 -0.16679268 0.50711435 -0.69758403 -0.11343016 -0.1992593 -0.19667974; -0.2698496 -0.6636745 0.38938048 -0.64329565 -0.16498129 0.25972456 0.2509788 -0.10913698; -0.72250265 -0.43049273 0.5355308 -0.48264045 -0.13910031 0.032667212 -0.44423008 0.21614327; 0.4879329 -0.18735494 -0.65818995 0.25029483 0.33856842 0.070783965 -0.5193983 -0.40827623; -0.42754745 0.35958946 0.24703826 0.3873787 -0.29482082 -0.18204269 -0.75290614 0.12673046; -0.6426762 0.5503079 0.27427903 -0.45793745 0.5987179 0.43792063 -0.97237444 0.8698215; -0.60542583 0.25829408 0.07128804 0.43198216 -0.25921226 0.56372166 -0.41576153 -0.30907118; -0.8200053 -0.27719298 0.45876926 -0.2748125 0.25648344 0.011891177 -0.8982788 0.7401685; 0.27366447 0.2731644 1.0160605 -0.915065 0.2561667 0.18906298 -1.1445975 0.5020349; -0.3430243 0.27395973 0.7715698 -0.4496679 0.44995722 0.89994836 -1.1919029 0.7052999; 0.4462397 0.3884464 -0.6883643 0.6586351 -1.1768486 -0.378989 -0.16866972 -0.06696147; -0.42308062 0.5155848 -0.24674061 0.74027264 -0.8131593 -0.108711556 0.08808072 -0.086081; -0.54075795 0.7455426 0.68438524 -0.5479886 0.63823694 0.06107664 -0.5104658 0.64088964], bias = Float32[0.28752583; 0.2684676; 0.14331727; 0.32473034; 0.34296423; 0.12274929; 0.007166392; 0.97074836; 0.37931964; 0.15817502; 0.010318942; 0.35584262; 0.43078434; 0.06631472; -0.012242681; 0.3143152; 0.8410905; 1.1331578; 1.1606256; 0.76865643; 0.68876714; 1.2339538; 0.6464516; 0.90202117; 0.4793701; 0.031714108; 0.29911605; 0.62638867; 0.6320951; 0.011824203; -0.11740463; 0.8846581;;]), classifier = (weight = Float32[-1.4402367 0.7417972 1.2297208 1.2689502 -0.93723375 0.11624937 -0.25733557 1.2114556], bias = Float32[-0.57882833;;])), (lstm_cell = (rng = Random.Xoshiro(0x2026f555c226bf09, 0x8a6bb764b93cadda, 0x5ba3c10439600514, 0x446f763658f71987),), classifier = NamedTuple()))
```


<a id='Saving-the-Model'></a>

## Saving the Model


We can save the model using JLD2 (and any other serialization library of your choice) Note that we transfer the model to CPU before saving. Additionally, we recommend that you don't save the model


```julia
@save "trained_model.jld2" {compress = true} ps_trained st_trained
```


Let's try loading the model


```julia
@load "trained_model.jld2" ps_trained st_trained
```


```
2-element Vector{Symbol}:
 :ps_trained
 :st_trained
```


---


*This page was generated using [Literate.jl](https://github.com/fredrikekre/Literate.jl).*


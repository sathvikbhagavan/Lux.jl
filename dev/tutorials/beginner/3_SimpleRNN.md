


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
((lstm_cell = (weight_i = Float32[-0.8604128 -0.42108482; -0.2533733 -0.75545484; 0.8029503 0.8828446; -0.054353494 0.075842254; -0.14514887 -0.62635124; 0.2905686 0.55332184; -0.83016586 0.06959652; 0.08173118 0.05325508; -0.041514337 0.08250012; -0.65977055 0.45117995; 1.1402394 -0.02323566; -0.008338998 0.088540204; -0.10299717 0.24794056; -0.8472596 0.32626617; -0.1314201 -0.27969053; 1.2138168 0.093606465; 0.6776514 -0.63052434; 0.13191317 -0.34816483; 0.47687525 -0.32806587; -0.5598016 0.5173987; 0.582341 -0.8404493; 0.10273888 0.29350585; 0.8658114 -0.64053154; 0.94441235 0.61418015; -1.2477576 -0.10795662; 0.49024746 0.64345384; 1.0556554 0.67896116; -0.47170237 -0.17707963; 0.7231422 -0.71031755; -0.33324188 0.75283086; -0.21580021 0.6133703; 0.6020973 -0.3182176], weight_h = Float32[-0.5077409 -0.05144826 0.28356618 -0.2591836 0.3015797 0.010954502 -0.6957032 0.55355585; -0.6471331 0.23570907 -0.06627596 0.63717353 0.39336398 0.31357104 0.13085212 -0.048466764; 0.019067425 0.059901997 -0.033740897 0.55679274 -0.7426493 0.32350847 -0.6324352 -0.17175017; 0.038874555 -0.27435473 0.8844464 -0.09389874 0.9109076 0.08106291 0.2786661 0.9066275; -0.37592643 0.47285876 0.7722334 0.2939763 0.40226403 0.642863 -0.35328948 -0.078917965; -0.05967953 -0.3383338 0.33312723 -0.09747203 -0.28880572 -0.22076127 -0.49771228 0.0473173; -0.7827133 0.36877826 0.6953079 0.49273902 -0.8022522 0.63889337 0.032992158 -0.6309041; 0.68356556 0.3061721 1.0212303 -1.3208938 0.82322425 0.16077457 -0.70661116 1.1456336; -0.14555496 0.47593927 0.40659025 -0.5059222 0.4609221 0.6793417 0.1909697 0.6376037; -0.41638163 -0.18871705 -0.37934917 -0.014881191 0.24824165 0.0018002347 -0.7504484 0.7570806; -0.14715096 0.5517258 -0.06273889 0.5270184 -0.52419835 0.29468173 -0.26757234 -0.38488972; 0.05756013 0.87834555 0.28317085 -0.1059255 0.82688385 0.61187804 0.39404693 0.49748287; 0.6508716 0.49525407 0.37098634 -0.06836757 0.9195485 0.19365928 -0.60978293 0.4402737; -0.105182365 0.44429755 0.08742545 0.07937327 0.61129916 -0.009354215 -0.1349286 -0.39994186; -0.8386093 0.3224924 0.09100826 -0.40630525 -0.25481024 0.8130452 -0.36406496 0.40998325; -0.644915 0.78677285 0.40800175 1.0364239 -0.09522319 0.82005024 -0.10797607 0.7591872; 0.1487644 -0.63632274 -0.049102765 -0.43222308 -0.35673922 -0.21546747 -0.16084154 0.18249276; -0.43638238 0.22699232 0.19937085 0.7123919 0.40199828 -0.3533354 -0.3643932 0.68213993; -0.55189264 0.70754445 0.09506867 -0.4342323 -0.3249342 0.73883194 0.052139506 0.5977596; -0.77500707 0.34008995 -0.16757199 0.502744 -0.6967847 -0.1683291 -0.19962852 -0.19805141; -0.272608 -0.66736615 0.40243948 -0.6423228 -0.17151953 0.2626806 0.24958125 -0.12183823; -0.7245348 -0.4261785 0.53093404 -0.47850493 -0.14685045 0.027665582 -0.44042763 0.220131; 0.48571742 -0.18876478 -0.6712584 0.30257758 0.33852273 0.066608995 -0.52238154 -0.4150697; -0.41176623 0.35334668 0.20769319 0.36689124 -0.3340237 -0.19712219 -0.7419776 0.013284476; -0.6462859 0.5582498 0.26787764 -0.45513973 0.5964691 0.44107965 -0.9442211 0.86691284; -0.60821456 0.28721192 0.062927745 0.4296515 -0.26493028 0.57083005 -0.41649452 -0.33023846; -0.822069 -0.27205676 0.4611476 -0.27139652 0.22915417 0.017775448 -0.88917845 0.7483984; 0.2975281 0.27059373 1.0165411 -0.8962193 0.23307946 0.186183 -1.0996706 0.4639047; -0.34064877 0.28330973 0.7754674 -0.42521837 0.44714907 0.9306723 -1.1852968 0.693835; 0.41740888 0.41339937 -0.6836431 0.6002481 -1.1387588 -0.35920995 -0.16798468 -0.049875576; -0.43608952 0.51947576 -0.25705636 0.69499946 -0.8189936 -0.1257205 0.07813891 -0.092323095; -0.54684204 0.76460505 0.70225835 -0.55741674 0.6145266 0.0651105 -0.4891348 0.6494047], bias = Float32[0.2947253; 0.2921762; 0.13731366; 0.32221305; 0.34438196; 0.11718227; 0.01075477; 0.9784984; 0.38407177; 0.1532954; 0.0064100525; 0.35483086; 0.43097848; 0.069220245; -0.013323414; 0.31327075; 0.8468737; 1.128072; 1.1544942; 0.76902384; 0.6895296; 1.2384961; 0.65693045; 0.92103624; 0.4778605; 0.038536727; 0.3041893; 0.61084855; 0.62886006; 0.034562793; -0.118886; 0.87743753;;]), classifier = (weight = Float32[-1.441288 0.76331383 1.2499055 1.2579887 -0.93540275 0.121890895 -0.26588246 1.2257926], bias = Float32[-0.58337784;;])), (lstm_cell = (rng = Random.Xoshiro(0x2026f555c226bf09, 0x8a6bb764b93cadda, 0x5ba3c10439600514, 0x446f763658f71987),), classifier = NamedTuple()))
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


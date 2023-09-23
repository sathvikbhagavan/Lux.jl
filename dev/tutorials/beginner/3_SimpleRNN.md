


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
((lstm_cell = (weight_i = Float32[-0.8629453 -0.41915968; -0.25351626 -0.7559784; 0.81337786 0.8831152; -0.059282746 0.07471742; -0.14594895 -0.6228577; 0.2871564 0.5544563; -0.8374499 0.07621; 0.07503707 0.052383736; -0.03935821 0.07158499; -0.6624349 0.45205364; 1.1650229 -0.017839337; -0.012417335 0.087646864; -0.11264788 0.24639066; -0.84999603 0.32058194; -0.12916525 -0.26960588; 1.2125492 0.1002233; 0.6799074 -0.6279804; 0.13026403 -0.34501356; 0.45761514 -0.32462674; -0.55431515 0.51871675; 0.5825276 -0.8373682; 0.09985541 0.2916752; 0.87172836 -0.6398239; 0.95363134 0.6278456; -1.2054089 -0.102335826; 0.48122492 0.64121896; 1.0553725 0.675306; -0.44144306 -0.16121757; 0.75239086 -0.6991748; -0.34211844 0.726422; -0.22029671 0.61035436; 0.61188567 -0.29053977], weight_h = Float32[-0.5103939 -0.05614025 0.29818293 -0.26573634 0.29564524 0.013192728 -0.71267885 0.5481209; -0.651843 0.21909901 -0.062311538 0.6341279 0.3808999 0.31036377 0.13172978 -0.04171901; 0.016538277 0.06023392 -0.034106918 0.56664765 -0.7399232 0.31362456 -0.64069325 -0.1533179; 0.03753527 -0.2681179 0.88912094 -0.09198877 0.9119003 0.09553567 0.27938294 0.90829927; -0.3666692 0.4702691 0.7843902 0.32268572 0.40257296 0.6362988 -0.3364736 -0.07603286; -0.06278615 -0.33515164 0.33163485 -0.09365001 -0.29025972 -0.21936858 -0.49853 0.039931778; -0.7909173 0.3740447 0.6894493 0.517707 -0.8132671 0.6381735 0.0249826 -0.62103665; 0.68866414 0.31191182 1.0763313 -1.3047721 0.8021636 0.16419636 -0.71532166 1.1403129; -0.16881251 0.47396222 0.42354953 -0.49637336 0.45759398 0.68212926 0.17189294 0.63493645; -0.41949007 -0.18638931 -0.3821914 -0.008821818 0.23971845 -0.00081064383 -0.745572 0.75918806; -0.15037636 0.55314195 -0.046427626 0.52958596 -0.51833415 0.29495537 -0.26997617 -0.37071538; 0.057080805 0.8842463 0.2878328 -0.10737909 0.8281615 0.64290035 0.3899416 0.49890107; 0.64919823 0.494397 0.374076 -0.06858928 0.9210256 0.1651261 -0.6690858 0.44228956; -0.09753288 0.4413287 0.088164695 0.08962908 0.6053206 -0.016536318 -0.14006631 -0.40273467; -0.8352584 0.32648736 0.08486887 -0.396378 -0.2643288 0.79880023 -0.37022308 0.4211398; -0.62861824 0.7894155 0.4110031 1.0363779 -0.09344845 0.8176814 -0.10052632 0.76254743; 0.15067632 -0.64004195 -0.057524715 -0.4412368 -0.35219264 -0.22293308 -0.16352573 0.17346135; -0.44019836 0.22932127 0.19773225 0.7164255 0.3921761 -0.35539874 -0.3612122 0.66915584; -0.5553643 0.7093773 0.0942553 -0.42998773 -0.31011868 0.74183863 0.049434178 0.6051528; -0.7773191 0.34008032 -0.16773756 0.50787944 -0.6990681 -0.13964655 -0.1995142 -0.19815916; -0.26834413 -0.6640654 0.39958304 -0.643099 -0.16382557 0.26336795 0.24851449 -0.118831605; -0.72954804 -0.423214 0.53410816 -0.47601673 -0.13772209 0.028682753 -0.44374633 0.21912421; 0.48608732 -0.16747041 -0.66202486 0.2519923 0.34134358 0.07332399 -0.5204515 -0.41987488; -0.41882676 0.3576571 0.2174014 0.37879768 -0.28136367 -0.18712547 -0.75165445 0.050362114; -0.64991707 0.5494161 0.26817796 -0.44593507 0.59811306 0.43385708 -0.9579004 0.8714585; -0.60400486 0.27285245 0.070163004 0.42863107 -0.26461405 0.5642855 -0.4145084 -0.29261506; -0.81782395 -0.27887422 0.46077654 -0.27094883 0.24037883 0.010087039 -0.8915804 0.74930614; 0.26694003 0.27552453 1.0192735 -0.8971446 0.24304375 0.178749 -1.1318184 0.48776478; -0.37062728 0.27592742 0.76927876 -0.4254118 0.4511142 0.8925491 -1.1906729 0.7166439; 0.4536374 0.38380316 -0.6908615 0.6322666 -1.2301576 -0.38294268 -0.14437114 -0.05795342; -0.43287638 0.51760757 -0.25587776 0.7195637 -0.81871223 -0.12773404 0.0882238 -0.08039092; -0.55445844 0.74174815 0.67840374 -0.5247454 0.65345347 0.054351773 -0.48960498 0.64083976], bias = Float32[0.28870347; 0.2766752; 0.1383063; 0.32429403; 0.34474298; 0.120242976; 0.010862419; 0.97034025; 0.3790547; 0.15595458; 0.008865204; 0.35680053; 0.4319764; 0.06672963; -0.013705398; 0.31540185; 0.84428525; 1.129769; 1.1579486; 0.76751995; 0.68977064; 1.2405021; 0.65753424; 0.9056595; 0.4761236; 0.049371693; 0.29808632; 0.6214211; 0.636752; 0.011474433; -0.11784494; 0.8785048;;]), classifier = (weight = Float32[-1.4330368 0.7531786 1.2362006 1.258496 -0.93438303 0.115839586 -0.26584885 1.2224594], bias = Float32[-0.63103646;;])), (lstm_cell = (rng = Random.Xoshiro(0x2026f555c226bf09, 0x8a6bb764b93cadda, 0x5ba3c10439600514, 0x446f763658f71987),), classifier = NamedTuple()))
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


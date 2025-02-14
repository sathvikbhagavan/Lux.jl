module LuxFluxTransformExt

import Flux

using Lux, Random, Optimisers
import Lux: transform, FluxLayer
import TruncatedStacktraces

struct FluxModelConversionError <: Exception
    msg::String
end

function Base.showerror(io::IO, e::FluxModelConversionError)
    print(io, "FluxModelConversionError(", e.msg, ")")
    if !TruncatedStacktraces.VERBOSE[]
        println(io, TruncatedStacktraces.VERBOSE_MSG)
    end
end

# Workaround https://github.com/cjdoris/PackageExtensionCompat.jl/issues/9
function __copy_anonymous_closure(x)
    function copy_anonymous_closure(args...)
        return x
    end
    return copy_anonymous_closure
end

function FluxLayer(l)
    p, re = Optimisers.destructure(l)
    p_ = copy(p)
    return FluxLayer(l, re, () -> p_)
end

Lux.initialparameters(::AbstractRNG, l::FluxLayer) = (p=l.init_parameters(),)

(l::FluxLayer)(x, ps, st) = l.re(ps.p)(x), st

Base.show(io::IO, l::FluxLayer) = print(io, "FluxLayer($(l.layer))")

function transform(l::T; preserve_ps_st::Bool=false, kwargs...) where {T}
    @warn "Transformation for type $T not implemented. Using `FluxLayer` as a fallback." maxlog=1

    if !preserve_ps_st
        @warn "`FluxLayer` uses the parameters and states of the `layer`. It is not possible to NOT preserve the parameters and states. Ignoring this keyword argument." maxlog=1
    end

    return FluxLayer(l)
end

transform(l::Function; kwargs...) = WrappedFunction(l)

function transform(l::Flux.Chain; kwargs...)
    fn = x -> transform(x; kwargs...)
    layers = map(fn, l.layers)
    if layers isa NamedTuple
        return Chain(layers; disable_optimizations=true)
    else
        return Chain(layers...; disable_optimizations=true)
    end
end

function transform(l::Flux.Dense; preserve_ps_st::Bool=false, kwargs...)
    out_dims, in_dims = size(l.weight)
    if preserve_ps_st
        bias = l.bias isa Bool ? nothing : reshape(copy(l.bias), out_dims, 1)
        return Dense(in_dims => out_dims, l.σ;
            init_weight=__copy_anonymous_closure(copy(l.weight)),
            init_bias=__copy_anonymous_closure(bias), use_bias=!(l.bias isa Bool))
    else
        return Dense(in_dims => out_dims, l.σ; use_bias=!(l.bias isa Bool))
    end
end

function transform(l::Flux.Scale; preserve_ps_st::Bool=false, kwargs...)
    if preserve_ps_st
        return Scale(size(l.scale), l.σ;
            init_weight=__copy_anonymous_closure(copy(l.scale)),
            init_bias=__copy_anonymous_closure(copy(l.bias)), use_bias=!(l.bias isa Bool))
    else
        return Scale(size(l.scale), l.σ; use_bias=!(l.bias isa Bool))
    end
end

transform(l::Flux.Maxout; kwargs...) = Maxout(transform.(l.layers; kwargs...)...)

function transform(l::Flux.SkipConnection; kwargs...)
    connection = l.connection isa Function ? l.connection :
                 transform(l.connection; kwargs...)
    return SkipConnection(transform(l.layers; kwargs...), connection)
end

function transform(l::Flux.Bilinear; preserve_ps_st::Bool=false, kwargs...)
    out, in1, in2 = size(l.weight)
    if preserve_ps_st
        return Bilinear((in1, in2) => out, l.σ;
            init_weight=__copy_anonymous_closure(copy(l.weight)),
            init_bias=__copy_anonymous_closure(copy(l.bias)), use_bias=!(l.bias isa Bool))
    else
        return Bilinear((in1, in2) => out, l.σ; use_bias=!(l.bias isa Bool))
    end
end

function transform(l::Flux.Parallel; kwargs...)
    fn = x -> transform(x; kwargs...)
    layers = map(fn, l.layers)
    if layers isa NamedTuple
        return Parallel(l.connection; layers...)
    else
        return Parallel(l.connection, layers...)
    end
end

function transform(l::Flux.PairwiseFusion; kwargs...)
    @warn "Flux.PairwiseFusion and Lux.PairwiseFusion are semantically different. Using `FluxLayer` as a fallback." maxlog=1
    return FluxLayer(l)
end

function transform(l::Flux.Embedding; preserve_ps_st::Bool=true, kwargs...)
    out_dims, in_dims = size(l.weight)
    if preserve_ps_st
        return Embedding(in_dims => out_dims;
            init_weight=__copy_anonymous_closure(copy(l.weight)))
    else
        return Embedding(in_dims => out_dims)
    end
end

function transform(l::Flux.Conv; preserve_ps_st::Bool=false, kwargs...)
    k = size(l.weight)[1:(end - 2)]
    in_chs, out_chs = size(l.weight)[(end - 1):end]
    groups = l.groups
    pad = l.pad isa Flux.SamePad ? SamePad() : l.pad
    if preserve_ps_st
        _bias = l.bias isa Bool ? nothing :
                reshape(copy(l.bias), ntuple(_ -> 1, length(k))..., out_chs, 1)
        return Conv(k, in_chs * groups => out_chs, l.σ; l.stride, pad, l.dilation, groups,
            init_weight=__copy_anonymous_closure(Lux._maybe_flip_conv_weight(l.weight)),
            init_bias=__copy_anonymous_closure(_bias), use_bias=!(l.bias isa Bool))
    else
        return Conv(k, in_chs * groups => out_chs, l.σ; l.stride, pad, l.dilation, groups,
            use_bias=!(l.bias isa Bool))
    end
end

function transform(l::Flux.ConvTranspose; preserve_ps_st::Bool=false, kwargs...)
    k = size(l.weight)[1:(end - 2)]
    out_chs, in_chs = size(l.weight)[(end - 1):end]
    groups = l.groups
    pad = l.pad isa Flux.SamePad ? SamePad() : l.pad
    if preserve_ps_st
        _bias = l.bias isa Bool ? nothing :
                reshape(copy(l.bias), ntuple(_ -> 1, length(k))..., out_chs, 1)
        return ConvTranspose(k, in_chs * groups => out_chs, l.σ; l.stride, pad, l.dilation,
            groups, use_bias=!(l.bias isa Bool),
            init_weight=__copy_anonymous_closure(Lux._maybe_flip_conv_weight(l.weight)),
            init_bias=__copy_anonymous_closure(_bias))
    else
        return ConvTranspose(k, in_chs * groups => out_chs, l.σ; l.stride, pad, l.dilation,
            groups, use_bias=!(l.bias isa Bool))
    end
end

function transform(l::Flux.CrossCor; preserve_ps_st::Bool=false, kwargs...)
    k = size(l.weight)[1:(end - 2)]
    in_chs, out_chs = size(l.weight)[(end - 1):end]
    pad = l.pad isa Flux.SamePad ? SamePad() : l.pad
    if preserve_ps_st
        _bias = l.bias isa Bool ? nothing :
                reshape(copy(l.bias), ntuple(_ -> 1, length(k))..., out_chs, 1)
        return CrossCor(k, in_chs => out_chs, l.σ; l.stride, pad, l.dilation,
            init_weight=__copy_anonymous_closure(copy(l.weight)),
            init_bias=__copy_anonymous_closure(_bias), use_bias=!(l.bias isa Bool))
    else
        return CrossCor(k, in_chs => out_chs, l.σ; l.stride, pad, l.dilation,
            use_bias=!(l.bias isa Bool))
    end
end

transform(l::Flux.AdaptiveMaxPool; kwargs...) = AdaptiveMaxPool(l.out)

transform(l::Flux.AdaptiveMeanPool; kwargs...) = AdaptiveMeanPool(l.out)

transform(::Flux.GlobalMaxPool; kwargs...) = GlobalMaxPool()

transform(::Flux.GlobalMeanPool; kwargs...) = GlobalMeanPool()

function transform(l::Flux.MaxPool; kwargs...)
    pad = l.pad isa Flux.SamePad ? SamePad() : l.pad
    return MaxPool(l.k; l.stride, pad)
end

function transform(l::Flux.MeanPool; kwargs...)
    pad = l.pad isa Flux.SamePad ? SamePad() : l.pad
    return MeanPool(l.k; l.stride, pad)
end

transform(l::Flux.Dropout; kwargs...) = Dropout(l.p; l.dims)

function transform(l::Flux.LayerNorm; kwargs...)
    @warn "Flux.LayerNorm and Lux.LayerNorm are semantically different specifications. Using `FluxLayer` as a fallback." maxlog=1
    return FluxLayer(l)
end

transform(::typeof(identity); kwargs...) = NoOpLayer()

transform(::typeof(Flux.flatten); kwargs...) = FlattenLayer()

transform(l::Flux.PixelShuffle; kwargs...) = PixelShuffle(l.r)

function transform(l::Flux.Upsample{mode}; kwargs...) where {mode}
    return Upsample(mode; l.scale, l.size)
end

function transform(l::Flux.RNNCell; preserve_ps_st::Bool=false, force_preserve::Bool=false)
    out_dims, in_dims = size(l.Wi)
    if preserve_ps_st
        if force_preserve
            throw(FluxModelConversionError("Recurrent Cell: $(typeof(l)) for Flux uses a `reset!` mechanism which hasn't been extensively tested with `FluxLayer`. Rewrite the model manually to use `RNNCell`."))
        end
        @warn "Preserving Parameters: `Wh` & `Wi` for `Flux.RNNCell` is ambiguous in Lux and hence not supported. Ignoring these parameters." maxlog=1
        return RNNCell(in_dims => out_dims, l.σ;
            init_bias=__copy_anonymous_closure(copy(l.b)),
            init_state=__copy_anonymous_closure(copy(l.state0)))
    else
        return RNNCell(in_dims => out_dims, l.σ)
    end
end

function transform(l::Flux.LSTMCell; preserve_ps_st::Bool=false, force_preserve::Bool=false)
    _out_dims, in_dims = size(l.Wi)
    out_dims = _out_dims ÷ 4
    if preserve_ps_st
        if force_preserve
            throw(FluxModelConversionError("Recurrent Cell: $(typeof(l)) for Flux uses a `reset!` mechanism which hasn't been extensively tested with `FluxLayer`. Rewrite the model manually to use `LSTMCell`."))
        end
        @warn "Preserving Parameters: `Wh` & `Wi` for `Flux.LSTMCell` is ambiguous in Lux and hence not supported. Ignoring these parameters." maxlog=1
        bs = Lux.multigate(l.b, Val(4))
        _s, _m = copy.(l.state0)
        return LSTMCell(in_dims => out_dims; init_bias=__copy_anonymous_closure.(bs),
            init_state=__copy_anonymous_closure(_s),
            init_memory=__copy_anonymous_closure(_m))
    else
        return LSTMCell(in_dims => out_dims)
    end
end

function transform(l::Flux.GRUCell; preserve_ps_st::Bool=false, force_preserve::Bool=false)
    _out_dims, in_dims = size(l.Wi)
    out_dims = _out_dims ÷ 3
    if preserve_ps_st
        if force_preserve
            throw(FluxModelConversionError("Recurrent Cell: $(typeof(l)) for Flux uses a `reset!` mechanism which hasn't been extensively tested with `FluxLayer`. Rewrite the model manually to use `GRUCell`."))
        end
        @warn "Preserving Parameters: `Wh` & `Wi` for `Flux.GRUCell` is ambiguous in Lux and hence not supported. Ignoring these parameters." maxlog=1
        bs = Lux.multigate(l.b, Val(3))
        return GRUCell(in_dims => out_dims; init_bias=_const_return_anon_function.(bs),
            init_state=__copy_anonymous_closure(copy(l.state0)))
    else
        return GRUCell(in_dims => out_dims)
    end
end

function transform(l::Flux.BatchNorm;
    preserve_ps_st::Bool=false,
    force_preserve::Bool=false)
    if preserve_ps_st
        if l.track_stats
            force_preserve && return FluxLayer(l)
            @warn "Preserving the state of `Flux.BatchNorm` is currently not supported. Ignoring the state." maxlog=1
        end
        if l.affine
            return BatchNorm(l.chs, l.λ; l.affine, l.track_stats, epsilon=l.ϵ, l.momentum,
                init_bias=__copy_anonymous_closure(copy(l.β)),
                init_scale=__copy_anonymous_closure(copy(l.γ)))
        else
            return BatchNorm(l.chs, l.λ; l.affine, l.track_stats, epsilon=l.ϵ, l.momentum)
        end
    end
    return BatchNorm(l.chs, l.λ; l.affine, l.track_stats, epsilon=l.ϵ, l.momentum)
end

function transform(l::Flux.GroupNorm;
    preserve_ps_st::Bool=false,
    force_preserve::Bool=false)
    if preserve_ps_st
        if l.track_stats
            force_preserve && return FluxLayer(l)
            @warn "Preserving the state of `Flux.GroupNorm` is currently not supported. Ignoring the state." maxlog=1
        end
        if l.affine
            return GroupNorm(l.chs, l.G, l.λ; l.affine, epsilon=l.ϵ,
                init_bias=__copy_anonymous_closure(copy(l.β)),
                init_scale=__copy_anonymous_closure(copy(l.γ)))
        else
            return GroupNorm(l.chs, l.G, l.λ; l.affine, epsilon=l.ϵ)
        end
    end
    return GroupNorm(l.chs, l.G, l.λ; l.affine, l.track_stats, epsilon=l.ϵ, l.momentum)
end

const _INVALID_TRANSFORMATION_TYPES = Union{<:Flux.Recur}

function transform(l::T; kwargs...) where {T <: _INVALID_TRANSFORMATION_TYPES}
    throw(FluxModelConversionError("Transformation of type $(T) is not supported."))
end

end

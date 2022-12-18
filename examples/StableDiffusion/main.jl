# Code based on https://github.com/geohot/tinygrad/blob/master/examples/stable_diffusion.py

using CUDA, CodecZlib, Images, LinearAlgebra, Lux, LuxLib, MLUtils, NNlib,
      OrderedCollections, Random, Setfield
using Lux: AbstractExplicitLayer, AbstractExplicitContainerLayer
import ChainRulesCore as CRC

CUDA.allowscalar(false)

silu(x) = x * sigmoid(x)

__reshape_back(y, x) = reshape(y, size(x))

__stable_cat(t::NTuple{N, T}, ::Val{dims}) where {dims, N, T} = cat(t...; dims)::T

function __normalize(x::AbstractArray{T}, ::Nothing) where {T}
    return layernorm(x, nothing, nothing; dims=(ndims(x) - 1), epsilon=convert(T, 1e-5))
end

function __normalize(x::AbstractArray{T}, num_groups) where {T}
    return reshape(layernorm(reshape(x, :, num_groups, size(x, ndims(x))), nothing, nothing;
                             dims=2, epsilon=convert(T, 1e-5)), size(x))
end

__batched_mul(x, y) = batched_mul(x, y)
function __batched_mul(x::AbstractArray{T1, 4}, y::AbstractArray{T2, 4}) where {T1, T2}
    x1, x2, x3, x4 = size(x)
    y1, y2, y3, y4 = size(y)
    @assert x3 == y3 && x4 == y4
    return reshape(__batched_mul(reshape(x, (x1, x2, x3 * x4)),
                                 reshape(y, (y1, y2, y3 * y4))), (x1, y2, x3, x4))
end

@inline __chunk(h::Int, n::Int) = (1:h) .+ h * (n - 1)
@inline function __fast_chunk(x::AbstractArray, h::Int, n::Int, ::Val{dim}) where {dim}
    return selectdim(x, dim, __chunk(h, n))
end
@inline function __fast_chunk(x::CuArray, h::Int, n::Int, ::Val{dim}) where {dim}
    return copy(selectdim(x, dim, __chunk(h, n)))
end
@inline function __fast_chunk(x::AbstractArray, ::Val{N}, d::Val{D}) where {N, D}
    return __fast_chunk.((x,), size(x, D) ÷ N, 1:N, d)
end

function Normalize(in_channels::Int, num_groups=32, activation=identity)
    lnorm = WrappedFunction(Base.Fix2(__normalize, num_groups))
    reshape_layer = WrappedFunction(x -> reshape(x, :, in_channels, size(x, ndims(x))))
    scale = Scale((1, in_channels), activation)
    skip_connection = SkipConnection(Chain(; reshape_layer, scale), __reshape_back)
    return Chain(; lnorm, skip_connection)
end

function __attention((q, k, v))
    W, H, C, B = size(q)

    q_ = permutedims(reshape(q, W * H, C, B), (2, 1, 3))
    k_ = reshape(k, W * H, C, B)
    w_ = softmax(batched_mul(k_, q_) .* convert(eltype(q), inv(sqrt(C))))

    v_ = reshape(v, W * H, C, B)
    w_ = permutedims(w_, (2, 1, 3))
    h_ = reshape(batched_mul(w_, v_), W, H, C, B)

    return h_
end

function AttentionBlock(in_channels)
    return SkipConnection(Chain(; norm=Normalize(in_channels),
                                qkv=BranchLayer(;
                                                q=Conv((1, 1), in_channels => in_channels),
                                                k=Conv((1, 1), in_channels => in_channels),
                                                v=Conv((1, 1), in_channels => in_channels)),
                                attention=WrappedFunction(__attention),
                                proj_out=Conv((1, 1), in_channels => in_channels)), +)
end

function ResNetBlock(in_channels, out_channels)
    shortcut = in_channels == out_channels ? NoOpLayer() :
               Conv((1, 1), in_channels => out_channels)
    return Parallel(+; shortcut,
                    chain=Chain(; norm_1=Normalize(in_channels, 32, swish),
                                conv_1=Conv((3, 3), in_channels => out_channels; pad=1),
                                norm_2=Normalize(out_channels, 32, swish),
                                conv_2=Conv((3, 3), out_channels => out_channels; pad=1)))
end

function Mid(block_in)
    return Chain(; block_1=ResNetBlock(block_in, block_in), attn_1=AttentionBlock(block_in),
                 block_2=ResNetBlock(block_in, block_in))
end

function Decoder()
    sizes = [(128, 256), (256, 512), (512, 512), (512, 512)]
    conv_in = Conv((3, 3), 4 => 512; pad=1)
    mid = Mid(512)

    layers = []
    for (i, s) in enumerate(reverse(sizes))
        block = Chain(; resnet_1=ResNetBlock(s[2], s[1]), resent_2=ResNetBlock(s[1], s[1]),
                      resnet_3=ResNetBlock(s[1], s[1]))
        if i != 1
            block = Chain(; block, upsample=Lux.Upsample(2),
                          c=Conv((3, 3), s[1] => s[1]; pad=1))
        end
        push!(layers, Symbol("block_$i") => block)
    end

    norm_out = Normalize(128, 32, swish)
    conv_out = Conv((3, 3), 128 => 3; pad=1)

    return Chain(; in_block=Chain(; conv_in, mid), main_block=Chain(; layers...),
                 out_block=Chain(; norm_out, conv_out))
end

function Encoder()
    sizes = [(128, 256), (256, 512), (512, 512), (512, 512)]
    conv_in = Conv((3, 3), 3 => 128; pad=1)

    layers = []
    for (i, s) in enumerate(sizes)
        block = Chain(; resnet_1=ResNetBlock(s[1], s[2]), resent_2=ResNetBlock(s[2], s[2]))
        if i != 4
            block = Chain(; block,
                          downsample=Conv((3, 3), s[2] => s[2]; stride=2, pad=(0, 1, 0, 1)))
        end
        push!(layers, Symbol("block_$i") => block)
    end

    mid = Mid(512)
    norm_out = Normalize(512, 32, swish)
    conv_out = Conv((3, 3), 512 => 3; pad=1)

    return Chain(; in_block=conv_in, main_block=Chain(; layers...),
                 out_block=Chain(; mid, norm_out, conv_out))
end

function AutoEncoderKL()
    return Chain(; encoder=Encoder(), quant_conv=Conv((1, 1), 8 => 8),
                 mean_extract=WrappedFunction(x -> x[:, :, 1:4, :]),
                 post_quant_conv=Conv((1, 1), 4 => 4), decoder=Decoder())
end

# not to be confused with ResnetBlock
function ResBlock(channels, emb_channels, out_channels)
    in_layers = Chain(; norm=Normalize(channels, 32, silu),
                      c=Conv((3, 3), channels => out_channels; pad=1))
    emb_layers = Chain(; act=WrappedFunction(Base.Fix1(broadcast, silu)),
                       linear=Dense(emb_channels => out_channels),
                       reshape_layer=WrappedFunction(x -> reshape(x, 1, 1, size(x)...)))
    out_layers = Chain(; norm=Normalize(out_channels, 32, silu),
                       c=Conv((3, 3), out_channels => out_channels; pad=1))
    skip_con = channels == out_channels ? NoOpLayer() :
               Conv((1, 1), channels => out_channels)
    return Chain(; input_restructure=WrappedFunction(((x, emb),) -> (x, (x, emb))),
                 block=Parallel(.+; skip_con,
                                main=Chain(; main=Parallel(.+; in_layers, emb_layers),
                                           out_layers)))
end

__conditional_replicate((x, c)::Tuple, v::Val) = __conditional_replicate(x, c, v)
__conditional_replicate(x, v::Val) = __conditional_replicate(x, nothing, v)
__conditional_replicate(x, ::Nothing, ::Val{N}) where {N} = (x, (x for _ in 1:N)...)
__conditional_replicate(x, c, ::Val{N}) where {N} = (x, (c for _ in 1:N)...)

function __cross_attention((q, k, v), (num_heads, head_size, scale))
    B = size(q, ndims(q))

    q_ = permutedims(reshape(q, head_size, num_heads, :, B), (1, 3, 2, 4)) # (head_size, time, num_heads, B)
    k_ = permutedims(reshape(k, head_size, num_heads, :, B), (3, 1, 2, 4)) # (time, head_size, num_heads, B)
    v_ = permutedims(reshape(v, head_size, num_heads, :, B), (1, 3, 2, 4)) # (head_size, time, num_heads, B)

    score = __batched_mul(k_, q_) .* convert(eltype(q_), scale)
    weights = softmax(score; dims=ndims(score))

    attention = permutedims(__batched_mul(v_, weights), (1, 3, 2, 4))    # (head_size, num_heads, time, B)

    return reshape(attention, head_size * num_heads, :, B)
end

function CrossAttention(query_dim, context_dim, n_heads, d_head)
    return Chain(;
                 correct_inputs=WrappedFunction(Base.Fix2(__conditional_replicate, Val(2))),
                 qkv=Parallel(nothing;
                              q=Dense(query_dim => n_heads * d_head; use_bias=false),
                              k=Dense(context_dim => n_heads * d_head; use_bias=false),
                              v=Dense(context_dim => n_heads * d_head; use_bias=false)),
                 cross_attention=WrappedFunction(Base.Fix2(__cross_attention,
                                                           (n_heads, d_head,
                                                            inv(sqrt(d_head))))),
                 to_out=Dense(n_heads * d_head => query_dim))
end

function __geglu_op(x)
    y, gate = __fast_chunk(x, Val(2), Val(1))
    return y .* gelu.(gate)
end

function GEGLU(dim_in, dim_out)
    return Chain(; proj=Dense(dim_in => dim_out * 2), g=WrappedFunction(__geglu_op))
end

FeedForward(dim, mult=4) = Chain(; g=GEGLU(dim, dim * mult), lin=Dense(dim * mult => dim))

function BasicTransformerBlock(dim, context_dim, n_heads, d_head)
    input_path = Parallel(nothing,
                          SkipConnection(Chain(; norm_1=Normalize(dim, nothing),
                                               attn_1=CrossAttention(dim, dim, n_heads,
                                                                     d_head)), +),
                          NoOpLayer())

    main_path = Chain(WrappedFunction(((x, c),) -> (x, x, c)),
                      Parallel(nothing, NoOpLayer(), Normalize(dim, nothing), NoOpLayer()),
                      WrappedFunction(((x, x_, c),) -> (x, (x_, c))),
                      Parallel(+, NoOpLayer(),
                               CrossAttention(dim, context_dim, n_heads, d_head)))

    output_path = SkipConnection(Chain(; norm_3=Normalize(dim, nothing),
                                       ff=FeedForward(dim)), +)

    return Chain(;
                 input_correction=WrappedFunction(Base.Fix2(__conditional_replicate,
                                                            Val(1))), input_path, main_path,
                 output_path)
end

function SpatialTransformer(channels, context_dim, n_heads, d_head)
    @assert channels == n_heads * d_head

    norm = Normalize(channels)
    proj_in = Conv((1, 1), channels => n_heads * d_head)
    transformer_block = BasicTransformerBlock(channels, context_dim, n_heads, d_head)
    proj_out = Conv((1, 1), n_heads * d_head => channels)

    function __reshape_permutedims((y, x, c))
        W, H, C, B = size(x)
        return (y, (W, H, C, B), (permutedims(reshape(x, W * H, C, B), (2, 1, 3)), c))
    end

    __reshape_inverse((y, sz, x)) = (y, reshape(permutedims(x, (2, 1, 3)), sz))

    return Chain(((x, c),) -> (x, x, c),
                 Parallel(nothing, NoOpLayer(), Chain(; norm, proj_in), NoOpLayer()),
                 WrappedFunction(__reshape_permutedims),
                 Parallel(nothing, NoOpLayer(), NoOpLayer(), transformer_block),
                 WrappedFunction(__reshape_inverse), Parallel(+, NoOpLayer(), proj_out))
end

Downsample(channels) = Conv((3, 3), channels => channels; stride=2, pad=1)

Upsample(channels) = Chain(Lux.Upsample(2), Conv((3, 3), channels => channels; pad=1))

__to_device(::AbstractArray, y) = cpu(y)
__to_device(::CuArray, y) = gpu(y)

function timestep_embedding(timesteps::AbstractVecOrMat, dim, max_period=10000)
    half = Float32(dim ÷ 2)
    freqs = Float32.(exp.(-log(max_period) .* (0.0f0:(half - 1)) ./ half))
    args = timesteps .* __to_device(timesteps, freqs)
    embedding = vcat(cos.(args), sin.(args))
    return reshape(embedding, :, size(embedding, 2))
end

Base.@kwdef struct UNetBlock{RB, ST, L} <:
                   AbstractExplicitContainerLayer{(:res_block, :spatial_transformer,
                                                   :layer)}
    res_block::RB = NoOpLayer()
    spatial_transformer::ST = NoOpLayer()
    layer::L = NoOpLayer()
end

__run_layer(::NoOpLayer, (x, y), ps, st) = x, st
__run_layer(l, x, ps, st) = l(x, ps, st)

function (ublock::UNetBlock)((x, emb, c), ps, st)
    x, st_1 = __run_layer(ublock.res_block, (x, emb), ps.res_block, st.res_block)
    x, st_2 = __run_layer(ublock.spatial_transformer, (x, context), ps.spatial_transformer,
                          st.spatial_transformer)
    x, st_3 = ublock.layer(x, ps.layer, st.layer)
    return (x, emb, c), (; res_block=st_1, spatial_transformer=st_2, layer=st_3)
end

function UNet()
    __cat((x, e, c), (y, _, _)) = __stable_cat((x, y), Val(3)), e, c
    time_embed = Chain(Base.Fix2(timestep_embedding, 320), Dense(320 => 1280, silu),
                       Dense(1280 => 1280))

    input_blocks = (UNetBlock(; layer=Conv((3, 3), 4 => 320; pad=1)),
                    UNetBlock(; res_block=ResBlock(320, 1280, 320),
                              spatial_transformer=SpatialTransformer(320, 768, 8, 40)),
                    UNetBlock(; res_block=ResBlock(320, 1280, 320),
                              spatial_transformer=SpatialTransformer(320, 768, 8, 40)),
                    UNetBlock(; layer=Downsample(320)),
                    UNetBlock(; res_block=ResBlock(320, 1280, 640),
                              spatial_transformer=SpatialTransformer(640, 768, 8, 80)),
                    UNetBlock(; res_block=ResBlock(640, 1280, 640),
                              spatial_transformer=SpatialTransformer(640, 768, 8, 80)),
                    UNetBlock(; layer=Downsample(640)),
                    UNetBlock(; res_block=ResBlock(640, 1280, 1280),
                              spatial_transformer=SpatialTransformer(1280, 768, 8, 160)),
                    UNetBlock(; res_block=ResBlock(1280, 1280, 1280),
                              spatial_transformer=SpatialTransformer(1280, 768, 8, 160)),
                    UNetBlock(; layer=Downsample(1280)),
                    UNetBlock(; res_block=ResBlock(1280, 1280, 1280)),
                    UNetBlock(; res_block=ResBlock(1280, 1280, 1280)))

    middle_block = Chain(UNetBlock(; res_block=ResBlock(1280, 1280, 1280),
                                   spatial_transformer=SpatialTransformer(1280, 768, 8,
                                                                          160)),
                         UNetBlock(; res_block=ResBlock(1280, 1280, 1280)))

    output_blocks = (UNetBlock(; res_block=ResBlock(2560, 1280, 1280)),
                     UNetBlock(; res_block=ResBlock(2560, 1280, 1280)),
                     UNetBlock(; res_block=ResBlock(2560, 1280, 1280),
                               layer=Upsample(1280)),
                     UNetBlock(; res_block=ResBlock(2560, 1280, 1280),
                               spatial_transformer=SpatialTransformer(1280, 768, 8, 160)),
                     UNetBlock(; res_block=ResBlock(2560, 1280, 1280),
                               spatial_transformer=SpatialTransformer(1280, 768, 8, 160)),
                     UNetBlock(; res_block=ResBlock(1920, 1280, 1280),
                               spatial_transformer=SpatialTransformer(1280, 768, 8, 160),
                               layer=Upsample(1280)),
                     UNetBlock(; res_block=ResBlock(1920, 1280, 640),
                               spatial_transformer=SpatialTransformer(640, 768, 8, 80)),
                     UNetBlock(; res_block=ResBlock(1280, 1280, 640),
                               spatial_transformer=SpatialTransformer(640, 768, 8, 80)),
                     UNetBlock(; res_block=ResBlock(960, 1280, 640),
                               spatial_transformer=SpatialTransformer(640, 768, 8, 80),
                               layer=Upsample(640)),
                     UNetBlock(; res_block=ResBlock(960, 1280, 320),
                               spatial_transformer=SpatialTransformer(320, 768, 8, 40)),
                     UNetBlock(; res_block=ResBlock(640, 1280, 320),
                               spatial_transformer=SpatialTransformer(320, 768, 8, 40)),
                     UNetBlock(; res_block=ResBlock(640, 1280, 320),
                               spatial_transformer=SpatialTransformer(320, 768, 8, 40)))

    chain = middle_block
    for (input_block, output_block) in zip(reverse(input_blocks), output_blocks)
        chain = Chain(; input_block, scon=SkipConnection(chain, __cat), output_block)
    end

    return Chain(; branch=Parallel(nothing; x=NoOpLayer(), time_embed, context=NoOpLayer()),
                 chain, drop=WrappedFunction(first),
                 output=Chain(Normalize(320, 32, silu), Conv((3, 3), 320 => 4; pad=1)))
end

CLIPMLP() = Chain(Dense(768 => 3072, gelu), Dense(3072 => 768))

function __clip_attention_reshape(x, seq_len, batch_size, num_heads, head_dim)
    return permutedims(reshape(x, (head_dim, num_heads, seq_len, batch_size)), (1, 3, 2, 4))
end

function __clip_attention((q, k, v, causal_attention_mask),
                          (num_heads, head_dim, scale, embed_dim))
    B = size(q, ndims(q))

    proj_shape = (head_dim, :, B * num_heads)

    q = reshape(__clip_attention_reshape(q .* convert(eltype(q), scale), :, B, num_heads,
                                         head_dim), proj_shape)
    k = reshape(__clip_attention_reshape(k, :, B, num_heads, head_dim), proj_shape)
    v = reshape(__clip_attention_reshape(v, :, B, num_heads, head_dim), proj_shape)

    src_len = size(k, 2)

    attn_weights = __batched_mul(permutedims(k, (2, 1, 3)), q)  # SL x SL x B
    attn_weights = reshape(attn_weights, (src_len, :, num_heads, B)) .+
                   causal_attention_mask
    attn_weights = softmax(reshape(attn_weights, (src_len, :, B * num_heads)))

    attn_output = reshape(permutedims(reshape(__batched_mul(v, attn_weights),
                                              (head_dim, :, num_heads, B)), (1, 3, 2, 4)),
                          (embed_dim, :, B))

    return attn_output
end

function CLIPAttention()
    embed_dim, num_heads = 768, 12
    head_dim = embed_dim ÷ num_heads
    return Chain(; input_restructure=WrappedFunction(((x, c),) -> (x, x, x, c)),
                 qkv=Parallel(nothing; q=Dense(embed_dim => embed_dim),
                              k=Dense(embed_dim => embed_dim),
                              v=Dense(embed_dim => embed_dim), noop=NoOpLayer()),
                 attention=WrappedFunction(Base.Fix2(__clip_attention,
                                                     (num_heads, head_dim,
                                                      inv(sqrt(head_dim)), embed_dim))),
                 output_proj=Dense(embed_dim => embed_dim))
end

function CLIPEncoderLayer()
    layer_norm = Normalize(768, nothing)
    block_1 = Chain(; input_restructure=WrappedFunction(((x, c),) -> (x, (x, c))),
                    res_block=Parallel(+; noop=NoOpLayer(),
                                       main_block=Chain(;
                                                        norm=Parallel(nothing; layer_norm,
                                                                      noop=NoOpLayer()),
                                                        self_attn=CLIPAttention())))

    return Chain(; block_1, block_2=SkipConnection(Chain(; layer_norm, mlp=CLIPMLP()), +))
end

function CLIPEncoder()
    __restructure = WrappedFunction(((x, c),) -> ((x, c), c))
    return Chain([Chain(__restructure, Parallel(nothing, CLIPEncoderLayer(), NoOpLayer()))
                  for _ in 1:12]...; disable_optimizations=true)
end

function CLIPTextEmbeddings()
    return Parallel(+; input=Embedding(49408 => 768), position=Embedding(77 => 768))
end

__get_position_ids(x::AbstractVector) = x, collect(1:length(x))
function __get_position_ids(x::CuVector)
    return x, gpu(collect(1:length(x)))::CuArray{Int64, 1, CUDA.Mem.DeviceBuffer}
end
function __get_position_ids(x::AbstractMatrix)
    return x, repeat(last(__get_position_ids(x[:, 1])); outer=(1, size(x, 2)))
end

CRC.@non_differentiable __get_position_ids(::Any)

function __get_causal_attention_mask(x)
    return (x,
            repeat(reshape(triu(ones_like(x, (77, 77)) .* convert(eltype(x), -Inf), 1),
                           (77, 77, 1, 1)); outer=(1, 1, 1, size(x, ndims(x)))))
end

function CLIPTextTransformer()
    return Chain(; input_reconstruction=WrappedFunction(__get_position_ids),
                 embeddings=CLIPTextEmbeddings(),
                 causal_attn_mask=WrappedFunction(__get_causal_attention_mask),
                 encoder=CLIPEncoder(), first_result=WrappedFunction(first),
                 final_layer_norm=Normalize(768, nothing))
end

# Clip tokenizer, taken from https://github.com/openai/CLIP/blob/main/clip/simple_tokenizer.py (MIT license)
default_bpe() = joinpath(@__DIR__, "assets/", "bpe_simple_vocab_16e6.txt.gz")

@inbounds @views function get_pairs(word)
    p = Set(((word[1], word[2]),))
    for i in 2:(length(word) - 1)
        union!(p, ((word[i], word[i + 1]),))
    end
    return p
end

whitespace_clean(text) = strip(replace(text, r"\s+" => " "))

function bytes_to_unicode()
    bs = vcat(codepoint('!'):codepoint('~'), codepoint('¡'):codepoint('¬'),
              codepoint('®'):codepoint('ÿ'))
    cs = copy(bs)

    n = 0
    for b in 0:(2^8 - 1)
        if b ∉ bs
            push!(bs, b)
            push!(cs, 2^8 + n)
            n += 1
        end
    end
    cs = Char.(cs)

    return OrderedDict(zip(bs, cs))
end

struct CLIPTokenizer{B, E, R, C, P}
    byte_encoder::B
    encoder::E
    bpe_ranks::R
    cache::C
    pattern::P
end

function CLIPTokenizer(bpe_path=default_bpe())
    byte_encoder = bytes_to_unicode()

    merges = readlines(GzipDecompressorStream(open(bpe_path, "r")))
    merges = Tuple.(split.(merges[2:(49152 - 256 - 2)]))::Vector{
                                                                 Tuple{SubString{String},
                                                                       SubString{String}}}

    vocab = string.(collect(values(byte_encoder)))
    vocab = vcat(vocab, vocab .* ("</w>",), join.(merges),
                 ["<|startoftext|>", "<|endoftext|>"])

    encoder = OrderedDict(zip(vocab, 1:length(vocab)))
    bpe_ranks = OrderedDict(zip(merges, 1:length(merges)))
    cache = OrderedDict("<|startoftext|>" => "<|startoftext|>",
                        "<|endoftext|>" => "<|endoftext|>")
    pattern = r"<\|startoftext\|>|<\|endoftext\|>|'s|'t|'re|'ve|'m|'ll|'d|[^\s]+"i

    return CLIPTokenizer(byte_encoder, encoder, bpe_ranks, cache, pattern)
end

function bpe!(tokenizer::CLIPTokenizer, token)
    token in keys(tokenizer.cache) && return tokenizer.cache[token]

    word = vcat(split(token[1:(end - 1)], ""), "$(token[end])</w>")
    pairs = get_pairs(word)

    length(pairs) == 0 && return "$(token)</w>"

    pairs = Tuple(pairs)
    while true
        _, idx = findmin(x -> get(tokenizer.bpe_ranks, x, typemax(Int)), pairs)
        bigram = pairs[idx]

        bigram ∉ keys(tokenizer.bpe_ranks) && break

        first, second = bigram
        new_word = []
        i = 1
        while i <= length(word)
            j = findnext(x -> x == first, word, i)
            if j !== nothing
                append!(new_word, word[i:(j - 1)])
                i = j
            else
                append!(new_word, word[i:end])
                break
            end

            if word[i] == first && i < length(word) && word[i + 1] == second
                push!(new_word, "$(first)$(second)")
                i += 2
            else
                push!(new_word, word[i])
                i += 1
            end
        end
        word = new_word
        if length(word) == 1
            break
        else
            pairs = Tuple(get_pairs(word))
        end
    end

    word = join(word, " ")
    tokenizer.cache[token] = word

    return word
end

function encode(tokenizer::CLIPTokenizer, text)
    bpe_tokens = Int64[]
    text = lowercase(whitespace_clean(text))
    for token in eachmatch(tokenizer.pattern, text)
        token = join(tokenizer.byte_encoder[UInt32(b)] for b in token.match)
        append!(bpe_tokens,
                [tokenizer.encoder[bpe_token]
                 for bpe_token in split(bpe!(tokenizer, token))])
    end
    if length(bpe_tokens) > 75
        bpe_tokens = bpe_tokens[1:75]
    end
    return vcat(49407, bpe_tokens, repeat([49408], 77 - length(bpe_tokens) - 1))
end

function get_model_output(diffusion_model, (latent, t, unconditional_context, context), ps,
                          st)
    timesteps = fill_like(latent, t, (1, size(latent, ndims(latent))))
    unconditional_latent, _ = diffusion_model((latent, timesteps, unconditional_context),
                                              ps, st)
    latent, _ = diffusion_model((latent, timesteps, context), ps, st)

    unconditional_guidance_scale = convert(eltype(latent), 7.5)
    e_t = unconditional_latent .+
          unconditional_guidance_scale .* (latent .- unconditional_latent)

    return e_t
end

function get_x_prev_and_pred_x0(x, e_t, index, αs, αs_prev)
    temperature = one(eltype(x))
    a_t, a_prev = αs[index], αs_prev[index]
    σ_t = zero(eltype(x))
    sqrt_one_minus_at = sqrt(1 - a_t)

    pred_x0 = (x .- sqrt_one_minus_at .* e_t) ./ sqrt(a_t)

    # direction pointing to x_t
    dir_xt = sqrt(1 - a_prev - σ_t^2) .* e_t
    noise = σ_t .* randn_like(x, size(x)) .* temperature

    x_prev = sqrt(a_prev) .* pred_x0 .+ dir_xt

    return x_prev, pred_x0
end

#=
steps = 5
αs_cumprod = rand(Float32, 1000)  # Load from file
batch_size = 1

tokenizer = CLIPTokenizer()
_text = "Picture of a horse sized cat eating a bagel"

clip = CLIPTextTransformer();
ps, st = Lux.setup(Xoshiro(0), clip) .|> gpu;

phrase_1 = encode(tokenizer, _text) |> gpu;
phrase_2 = encode(tokenizer, "") |> gpu;
phrase = hcat(phrase_1, phrase_2);

res, _ = clip(phrase, ps, st);
context, unconditional_context = res[:, :, 1:1], res[:, :, 2:2]
@info "Context vector generated."

timesteps = Int64.(range(1, 1000, step=1000÷steps))
@info "Running for" timesteps
αs = αs_cumprod[timesteps]
αs_prev = vcat(one(eltype(αs)), αs[1:(end - 1)])

diffusion_model = UNet();
ps, st = Lux.setup(Xoshiro(0), diffusion_model) .|> gpu;

latent = randn(Float32, 64, 64, 4, batch_size) |> gpu;
context = context |> gpu;
unconditional_context = unconditional_context |> gpu;

# this is diffusion
for (index, timestep) in enumerate(reverse(timesteps))
    global latent

    @info "Index = $index Timestep = $timestep"
    e_t = get_model_output(diffusion_model, (latent, timestep, unconditional_context,
                                             context), ps, st)
    x_prev, pred_x0 = get_x_prev_and_pred_x0(latent, e_t, index, αs, αs_prev)
    latent = x_prev
end

# upsample latent space to image with autoencoder
post_quant_conv = Conv((1, 1), 4 => 4);
ps, st = Lux.setup(Xoshiro(0), post_quant_conv) .|> gpu;

x_, _ = post_quant_conv(latent, ps, st);

decoder = Decoder();
ps, st = Lux.setup(Xoshiro(0), decoder) .|> gpu;

x_, _ = decoder(x_, ps, st);

# make image correct size and scale
x = clamp.(permutedims(reshape((x_ .+ 1.0f0) ./ 2.0f0, (512, 512, 3)), (3, 1, 2)), 0, 1);
@info "size(x) = $(size(x))"

# save image
img = colorview(RGB, x)
save("stable_diffusion_output.jpg", img)
=#

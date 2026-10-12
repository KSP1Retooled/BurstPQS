using BurstPQS.Util.Textures;
using Unity.Burst;
using Unity.Collections;
using Unity.Collections.LowLevel.Unsafe;
using Unity.Jobs;
using Unity.Mathematics;
using UnityEngine;

namespace BurstPQS.Jobs;

internal enum BlockEncoding
{
    BC1,
    BC3,

    /// <summary>
    /// BC3 holding a tangent-space normal the way KSP stores its own: Y in RGB, X in alpha.
    /// </summary>
    DXT5nm,

    /// <summary>BC5 with X in red and Y in green.</summary>
    BC5,
    BC7,
    BC7Fast,
}

internal static class BlockEncodingExt
{
    public static int BytesPerBlock(this BlockEncoding encoding) =>
        encoding == BlockEncoding.BC1 ? 8 : 16;
}

/// <summary>
/// Compresses an RGBA texture into 4x4 blocks. Each batch element is one block, numbered
/// row-major. Both dimensions must be multiples of 4.
/// </summary>
[BurstCompile]
internal unsafe struct TextureExportCompressJob : IJobParallelForBatch
{
    [ReadOnly]
    public NativeArray<Color32> pixels;

    /// <summary><see cref="BlockEncodingExt.BytesPerBlock"/> bytes per block.</summary>
    [WriteOnly]
    [NativeDisableParallelForRestriction]
    public NativeArray<byte> output;

    public BlockEncoding encoding;
    public int resX;

    public void Execute(int start, int count)
    {
        var px = stackalloc float4[16];
        var channel = stackalloc float[16];
        var dst = (byte*)NativeArrayUnsafeUtility.GetUnsafeBufferPointerWithoutChecks(output);
        int end = start + count;

        for (int block = start; block < end; block++)
        {
            LoadBlock(pixels, resX, block, px);

            switch (encoding)
            {
                case BlockEncoding.BC1:
                    ((ulong*)dst)[block] = BC1.Encode(px);
                    break;

                case BlockEncoding.BC3:
                    for (int i = 0; i < 16; i++)
                        channel[i] = px[i].w;

                    ((Block128*)dst)[block] = new Block128
                    {
                        lo = BC4.Encode(channel),
                        hi = BC1.Encode(px),
                    };
                    break;

                case BlockEncoding.DXT5nm:
                    for (int i = 0; i < 16; i++)
                        channel[i] = px[i].x;
                    ulong alpha = BC4.Encode(channel);

                    for (int i = 0; i < 16; i++)
                        channel[i] = px[i].y;

                    ((Block128*)dst)[block] = new Block128
                    {
                        lo = alpha,
                        hi = BC1.EncodeGrey(channel),
                    };
                    break;

                case BlockEncoding.BC5:
                    for (int i = 0; i < 16; i++)
                        channel[i] = px[i].x;
                    ulong red = BC4.Encode(channel);

                    for (int i = 0; i < 16; i++)
                        channel[i] = px[i].y;

                    ((Block128*)dst)[block] = new Block128 { lo = red, hi = BC4.Encode(channel) };
                    break;

                case BlockEncoding.BC7:
                case BlockEncoding.BC7Fast:
                    ((Block128*)dst)[block] = BC7.Encode(px, encoding == BlockEncoding.BC7Fast);
                    break;
            }
        }
    }

    static void LoadBlock(NativeArray<Color32> pixels, int resX, int block, float4* dst)
    {
        int blocksX = resX / 4;
        int x0 = block % blocksX * 4;
        int y0 = block / blocksX * 4;

        for (int y = 0; y < 4; y++)
        {
            int row = (y0 + y) * resX + x0;
            for (int x = 0; x < 4; x++)
            {
                var c = pixels[row + x];
                dst[y * 4 + x] = new float4(c.r, c.g, c.b, c.a);
            }
        }
    }
}

/// <summary>
/// Encodes heights to BC4, normalized to [0, 1] between <see cref="minH"/> and
/// <see cref="maxH"/>. Each batch element is one 4x4 block, numbered row-major.
/// </summary>
[BurstCompile]
internal unsafe struct TextureExportEncodeHeightsBC4Job : IJobParallelForBatch
{
    [ReadOnly]
    public NativeArray<float> heights;

    [WriteOnly]
    public NativeArray<ulong> output;

    public float minH;
    public float maxH;
    public int resX;

    public void Execute(int start, int count)
    {
        var values = stackalloc float[16];
        int blocksX = resX / 4;
        int end = start + count;

        float range = maxH - minH;
        float scale = range > 0f && math.isfinite(range) ? 255f / range : 0f;

        for (int block = start; block < end; block++)
        {
            int x0 = block % blocksX * 4;
            int y0 = block / blocksX * 4;

            for (int y = 0; y < 4; y++)
            {
                int row = (y0 + y) * resX + x0;
                for (int x = 0; x < 4; x++)
                    values[y * 4 + x] = math.clamp((heights[row + x] - minH) * scale, 0f, 255f);
            }

            output[block] = BC4.Encode(values);
        }
    }
}

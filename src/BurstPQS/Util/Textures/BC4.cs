using Unity.Mathematics;

namespace BurstPQS.Util.Textures;

internal static unsafe class BC4
{
    /// <summary>
    /// Encodes 16 values in [0, 255] as one BC4 block.
    /// </summary>
    /// <remarks>
    /// A block reaching 0 or 255 also tries the 6-value mode, whose palette holds both
    /// extremes exactly and leaves its endpoints free to cover the rest. This is what texconv
    /// does, except that texconv uses the 6-value mode unconditionally for such blocks.
    /// </remarks>
    public static ulong Encode(float* v)
    {
        float lo = v[0];
        float hi = v[0];
        for (int i = 1; i < 16; i++)
        {
            lo = math.min(lo, v[i]);
            hi = math.max(hi, v[i]);
        }

        int e0 = (int)math.round(hi);
        int e1 = (int)math.round(lo);
        if (e0 == e1)
            return Pack(e0, e1, 0);

        ulong best = Refine(v, e0, e1, out float bestError);

        if (lo < 0.5f || hi > 254.5f)
        {
            float innerLo = 255f;
            float innerHi = 0f;
            for (int i = 0; i < 16; i++)
            {
                if (v[i] < 0.5f || v[i] > 254.5f)
                    continue;

                innerLo = math.min(innerLo, v[i]);
                innerHi = math.max(innerHi, v[i]);
            }

            if (innerLo > innerHi)
                innerLo = innerHi = 0f;

            ulong six = Refine(
                v,
                (int)math.round(innerLo),
                (int)math.round(innerHi),
                out float sixError
            );
            if (sixError < bestError)
                best = six;
        }

        return best;
    }

    // Least-squares passes over the interpolated entries, keeping to whichever mode the
    // starting endpoints select: e0 > e1 for 8 values, e0 <= e1 for 6 plus 0 and 255.
    static ulong Refine(float* v, int e0, int e1, out float bestError)
    {
        bool eightValue = e0 > e1;

        bestError = Fit(v, e0, e1, out ulong bestIndices);
        int best0 = e0;
        int best1 = e1;

        for (int iter = 0; iter < BlockMath.RefineIterations; iter++)
        {
            float a = 0f,
                b = 0f,
                c = 0f,
                d0 = 0f,
                d1 = 0f;
            for (int i = 0; i < 16; i++)
            {
                float w = Weight((int)(bestIndices >> (i * 3)) & 7, eightValue);
                if (w < 0f)
                    continue;

                float u = 1f - w;
                a += u * u;
                b += u * w;
                c += w * w;
                d0 += u * v[i];
                d1 += w * v[i];
            }

            float det = a * c - b * b;
            if (math.abs(det) < 1e-6f)
                break;

            float inv = 1f / det;
            e0 = math.clamp((int)math.round((c * d0 - b * d1) * inv), 0, 255);
            e1 = math.clamp((int)math.round((a * d1 - b * d0) * inv), 0, 255);

            if (eightValue ? e0 < e1 : e0 > e1)
                (e0, e1) = (e1, e0);
            if (eightValue && e0 == e1)
                break;

            float error = Fit(v, e0, e1, out ulong indices);
            if (error >= bestError)
                break;

            bestError = error;
            best0 = e0;
            best1 = e1;
            bestIndices = indices;
        }

        return Pack(best0, best1, bestIndices);
    }

    // The weight of e1 at each index, or -1 for the 6-value mode's fixed 0 and 255.
    static float Weight(int index, bool eightValue) =>
        index switch
        {
            0 => 0f,
            1 => 1f,
            _ when eightValue => (index - 1) / 7f,
            >= 6 => -1f,
            _ => (index - 1) / 5f,
        };

    static float Fit(float* v, int e0, int e1, out ulong indices)
    {
        var palette = stackalloc float[8];
        palette[0] = e0;
        palette[1] = e1;

        if (e0 > e1)
        {
            for (int i = 1; i < 7; i++)
                palette[i + 1] = ((7 - i) * e0 + i * e1) / 7f;
        }
        else
        {
            for (int i = 1; i < 5; i++)
                palette[i + 1] = ((5 - i) * e0 + i * e1) / 5f;
            palette[6] = 0f;
            palette[7] = 255f;
        }

        float error = 0f;
        indices = 0;

        for (int i = 0; i < 16; i++)
        {
            float min = float.PositiveInfinity;
            ulong index = 0;

            for (int j = 0; j < 8; j++)
            {
                float d = palette[j] - v[i];
                d *= d;
                if (d < min)
                {
                    min = d;
                    index = (ulong)j;
                }
            }

            error += min;
            indices |= index << (i * 3);
        }

        return error;
    }

    static ulong Pack(int e0, int e1, ulong indices) =>
        (uint)e0 | (ulong)(uint)e1 << 8 | indices << 16;
}

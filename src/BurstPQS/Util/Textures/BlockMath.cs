using Unity.Mathematics;

namespace BurstPQS.Util.Textures;

/// <summary>
/// Shared pieces of the block encoders. Every one of them fits endpoints along the block's
/// principal axis, then refines them with a couple of least-squares passes over the chosen
/// indices.
/// </summary>
internal static class BlockMath
{
    public const int RefineIterations = 2;

    // Power iteration. A block with no spread falls back to the grey axis, which still puts
    // both endpoints on the block's one colour.
    public static float3 PrincipalAxis(float3x3 cov)
    {
        float3 axis = new(1f, 1f, 1f);
        for (int i = 0; i < 8; i++)
        {
            axis = math.mul(cov, axis);
            float len = math.length(axis);
            if (len < 1e-6f)
                return math.normalize(new float3(1f, 1f, 1f));
            axis /= len;
        }
        return axis;
    }

    public static float4 PrincipalAxis(float4x4 cov)
    {
        float4 axis = new(1f, 1f, 1f, 1f);
        for (int i = 0; i < 8; i++)
        {
            axis = math.mul(cov, axis);
            float len = math.length(axis);
            if (len < 1e-6f)
                return new float4(0.5f);
            axis /= len;
        }
        return axis;
    }
}

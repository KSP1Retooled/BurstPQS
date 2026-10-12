using System;
using System.Diagnostics;
using System.IO;
using System.Text;
using UnityEditor;
using UnityEngine;

// Encodes a PNG with BC7Encode.compute, decodes the result on the GPU and reports how close it
// is to the source. Only the top mip level, since that's all the kernel sees per dispatch.
public class BC7TestWindow : EditorWindow
{
    // Matches the exporter, which keeps each dispatch well under the driver's timeout.
    const int BlocksPerDispatch = 65536;
    const int BlocksPerGroup = 8;

    ComputeShader shader;
    string sourcePath = "";
    bool fast;
    bool saveDds = true;

    string result = "";
    Texture2D decoded;

    [MenuItem("BurstPQS/BC7 Test")]
    static void Open() => GetWindow<BC7TestWindow>("BC7 Test");

    void OnEnable()
    {
        if (shader == null)
            shader = AssetDatabase.LoadAssetAtPath<ComputeShader>("Assets/Shader/BC7Encode.compute");
    }

    void OnDisable()
    {
        if (decoded != null)
            DestroyImmediate(decoded);
    }

    void OnGUI()
    {
        shader = (ComputeShader)EditorGUILayout.ObjectField("Shader", shader, typeof(ComputeShader), false);

        using (new EditorGUILayout.HorizontalScope())
        {
            sourcePath = EditorGUILayout.TextField("Source PNG", sourcePath);
            if (GUILayout.Button("...", GUILayout.Width(30)))
            {
                var dir = File.Exists(sourcePath) ? Path.GetDirectoryName(sourcePath) : "";
                var picked = EditorUtility.OpenFilePanel("Source image", dir, "png");
                if (!string.IsNullOrEmpty(picked))
                    sourcePath = picked;
            }
        }

        fast = EditorGUILayout.Toggle("Fast", fast);
        saveDds = EditorGUILayout.Toggle("Save DDS", saveDds);

        using (new EditorGUI.DisabledScope(shader == null || !File.Exists(sourcePath)))
        {
            if (GUILayout.Button("Encode"))
                Run();
        }

        if (result.Length > 0)
            EditorGUILayout.HelpBox(result, MessageType.None);

        if (decoded != null)
        {
            var rect = GUILayoutUtility.GetAspectRect((float)decoded.width / decoded.height);
            // The pixels are stored top row first, which Unity draws upside down.
            GUI.DrawTextureWithTexCoords(rect, decoded, new Rect(0, 1, 1, -1));
        }
    }

    void Run()
    {
        var pixels = LoadTopDown(sourcePath, out int width, out int height);
        int blocksX = (width + 3) / 4;
        int blocksY = (height + 3) / 4;

        var blocks = Encode(pixels, width, height, out double ms);
        var decodedPixels = Decode(blocks, blocksX, blocksY);

        var sb = new StringBuilder();
        sb.AppendLine($"{width}x{height}, {blocksX * blocksY} blocks, {(fast ? "fast" : "full")}");
        sb.AppendLine($"Dispatch + readback: {ms:F1} ms (the first run includes pipeline creation)");
        AppendPsnr(sb, pixels, decodedPixels, width, height, blocksX * 4);
        AppendModes(sb, blocks);

        if (saveDds)
        {
            var path = Path.Combine(
                Path.GetDirectoryName(sourcePath),
                Path.GetFileNameWithoutExtension(sourcePath) + (fast ? "_gpu_fast.dds" : "_gpu.dds")
            );
            WriteDds(path, blocks, width, height);
            sb.AppendLine($"Wrote {path}");
        }

        result = sb.ToString().TrimEnd();
        UnityEngine.Debug.Log("[BC7Test] " + result);
    }

    // Rows top first, the way the exporter hands them to the encoder.
    static Color32[] LoadTopDown(string path, out int width, out int height)
    {
        var tex = new Texture2D(2, 2, TextureFormat.RGBA32, false, true);
        try
        {
            tex.LoadImage(File.ReadAllBytes(path));
            width = tex.width;
            height = tex.height;

            var bottomUp = tex.GetPixels32();
            var pixels = new Color32[bottomUp.Length];
            for (int y = 0; y < height; y++)
                Array.Copy(bottomUp, (height - 1 - y) * width, pixels, y * width, width);
            return pixels;
        }
        finally
        {
            DestroyImmediate(tex);
        }
    }

    byte[] Encode(Color32[] pixels, int width, int height, out double ms)
    {
        int kernel = shader.FindKernel(fast ? "EncodeBC7Fast" : "EncodeBC7");
        int blocksX = (width + 3) / 4;
        int blocksY = (height + 3) / 4;

        // The whole image goes up at once, so each dispatch's slot is just where its rows
        // already sit.
        var pixelBuffer = new ComputeBuffer(pixels.Length, 4);
        var blockBuffer = new ComputeBuffer(blocksX * blocksY, 16);
        try
        {
            shader.SetBuffer(kernel, "_Pixels", pixelBuffer);
            shader.SetBuffer(kernel, "_Blocks", blockBuffer);
            shader.SetInt("_ResX", width);
            shader.SetInt("_ResY", height);

            var words = new uint[blocksX * blocksY * 4];
            var timer = Stopwatch.StartNew();
            pixelBuffer.SetData(pixels);

            int rowsPerDispatch = Math.Max(1, BlocksPerDispatch / blocksX);
            for (int row = 0; row < blocksY; row += rowsPerDispatch)
            {
                int end = Math.Min(row + rowsPerDispatch, blocksY);
                shader.SetInt("_BlockRowOffset", row);
                shader.SetInt("_BlockRowEnd", end);
                shader.SetInt("_PixelOffset", row * 4 * width);
                shader.SetInt("_BlockOffset", row * blocksX);
                if (fast)
                    shader.Dispatch(kernel, (blocksX + 7) / 8, (end - row + 7) / 8, 1);
                else
                    shader.Dispatch(kernel, ((end - row) * blocksX + BlocksPerGroup - 1) / BlocksPerGroup, 1, 1);
            }

            blockBuffer.GetData(words);
            ms = timer.Elapsed.TotalMilliseconds;

            var bytes = new byte[words.Length * 4];
            Buffer.BlockCopy(words, 0, bytes, 0, bytes.Length);
            return bytes;
        }
        finally
        {
            pixelBuffer.Release();
            blockBuffer.Release();
        }
    }

    // Decodes at the padded size, since a BC7 texture has to be whole blocks.
    Color32[] Decode(byte[] blocks, int blocksX, int blocksY)
    {
        int w = blocksX * 4;
        int h = blocksY * 4;

        var bc7 = new Texture2D(w, h, TextureFormat.BC7, false, true);
        var rt = RenderTexture.GetTemporary(w, h, 0, RenderTextureFormat.ARGB32, RenderTextureReadWrite.Linear);
        var previous = RenderTexture.active;
        try
        {
            bc7.LoadRawTextureData(blocks);
            bc7.Apply(false);
            Graphics.Blit(bc7, rt);

            if (decoded != null)
                DestroyImmediate(decoded);
            decoded = new Texture2D(w, h, TextureFormat.RGBA32, false, true);

            RenderTexture.active = rt;
            decoded.ReadPixels(new Rect(0, 0, w, h), 0, 0);
            decoded.Apply(false);
            return decoded.GetPixels32();
        }
        finally
        {
            RenderTexture.active = previous;
            RenderTexture.ReleaseTemporary(rt);
            DestroyImmediate(bc7);
        }
    }

    static void AppendPsnr(StringBuilder sb, Color32[] source, Color32[] decoded, int width, int height, int decodedWidth)
    {
        var sse = new double[4];
        for (int y = 0; y < height; y++)
        {
            for (int x = 0; x < width; x++)
            {
                var a = source[y * width + x];
                var b = decoded[y * decodedWidth + x];
                sse[0] += Sq(a.r - b.r);
                sse[1] += Sq(a.g - b.g);
                sse[2] += Sq(a.b - b.b);
                sse[3] += Sq(a.a - b.a);
            }
        }

        double n = (double)width * height;
        double rgb = Psnr((sse[0] + sse[1] + sse[2]) / (3 * n));
        sb.AppendLine(
            $"PSNR RGB {rgb:F2} dB (R {Psnr(sse[0] / n):F2}, G {Psnr(sse[1] / n):F2}, B {Psnr(sse[2] / n):F2}), A {Psnr(sse[3] / n):F2} dB"
        );
    }

    static double Sq(int v) => (double)v * v;

    static double Psnr(double mse) => mse <= 0 ? double.PositiveInfinity : 10 * Math.Log10(255.0 * 255.0 / mse);

    // A block's mode is the number of zero bits before its first set bit.
    static void AppendModes(StringBuilder sb, byte[] blocks)
    {
        var counts = new int[9];
        for (int i = 0; i < blocks.Length; i += 16)
        {
            int first = blocks[i];
            int mode = 0;
            while (mode < 8 && (first & (1 << mode)) == 0)
                mode++;
            counts[mode]++;
        }

        sb.Append("Modes:");
        for (int mode = 0; mode < 8; mode++)
        {
            if (counts[mode] > 0)
                sb.Append($" {mode}: {counts[mode]}");
        }
        if (counts[8] > 0)
            sb.Append($" invalid: {counts[8]}");
        sb.AppendLine();
    }

    static void WriteDds(string path, byte[] blocks, int width, int height)
    {
        using (var writer = new BinaryWriter(File.Create(path)))
        {
            writer.Write(0x20534444u); // "DDS "
            writer.Write(124u);
            writer.Write(0x1u | 0x2u | 0x4u | 0x1000u | 0x80000u); // caps, height, width, pixel format, linear size
            writer.Write((uint)height);
            writer.Write((uint)width);
            writer.Write((uint)blocks.Length);
            writer.Write(0u); // depth
            writer.Write(1u); // mip count
            for (int i = 0; i < 11; i++)
                writer.Write(0u);

            writer.Write(32u);
            writer.Write(0x4u); // FourCC
            writer.Write(0x30315844u); // "DX10"
            for (int i = 0; i < 5; i++)
                writer.Write(0u);

            writer.Write(0x1000u); // texture
            for (int i = 0; i < 4; i++)
                writer.Write(0u);

            writer.Write(98u); // DXGI_FORMAT_BC7_UNORM
            writer.Write(3u); // texture 2D
            writer.Write(0u);
            writer.Write(1u); // array size
            writer.Write(0u);

            writer.Write(blocks);
        }
    }
}

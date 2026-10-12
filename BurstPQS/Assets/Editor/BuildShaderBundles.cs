using System.IO;
using UnityEditor;
using UnityEngine;
using UnityEngine.Rendering;

// Builds the shader bundles the mod loads, into src/GameData/Shaders so the mod's build copies
// them into GameData. Platforms without build support installed are skipped. Can also be run
// with -executeMethod BuildShaderBundles.Build.
public static class BuildShaderBundles
{
    static readonly string[] Assets = { "Assets/Shader/BC7Encode.compute" };

    [MenuItem("BurstPQS/Build Shader Bundles")]
    public static void Build()
    {
        var outDir = Path.GetFullPath(
            Path.Combine(Application.dataPath, "..", "..", "src", "GameData", "Shaders")
        );
        Directory.CreateDirectory(outDir);

        // Compute shaders are only compiled for the APIs in the target's player settings, and
        // the defaults leave out ones KSP can be forced onto, like D3D12 on Windows.
        var targets = new[]
        {
            (
                target: BuildTarget.StandaloneWindows64,
                name: "windows",
                apis: new[]
                {
                    GraphicsDeviceType.Direct3D11,
                    GraphicsDeviceType.Direct3D12,
                    GraphicsDeviceType.Vulkan,
                    GraphicsDeviceType.OpenGLCore,
                }
            ),
            (
                target: BuildTarget.StandaloneLinux64,
                name: "linux",
                apis: new[] { GraphicsDeviceType.OpenGLCore, GraphicsDeviceType.Vulkan }
            ),
            (
                target: BuildTarget.StandaloneOSX,
                name: "macos",
                apis: new[] { GraphicsDeviceType.Metal }
            ),
        };

        foreach (var (target, name, apis) in targets)
        {
            if (!BuildPipeline.IsBuildTargetSupported(BuildTargetGroup.Standalone, target))
            {
                Debug.LogWarning($"[BuildShaderBundles] No build support for {target}, skipping it");
                continue;
            }

            var bundle = new AssetBundleBuild
            {
                assetBundleName = $"burstpqs-{name}.unity3d",
                assetNames = Assets,
            };

            bool useDefaultApis = PlayerSettings.GetUseDefaultGraphicsAPIs(target);
            var previousApis = PlayerSettings.GetGraphicsAPIs(target);
            try
            {
                PlayerSettings.SetUseDefaultGraphicsAPIs(target, false);
                PlayerSettings.SetGraphicsAPIs(target, apis);

                // The build also writes manifests, which the mod doesn't need.
                var buildDir = Path.Combine("Temp", "ShaderBundles", name);
                Directory.CreateDirectory(buildDir);
                BuildPipeline.BuildAssetBundles(
                    buildDir,
                    new[] { bundle },
                    BuildAssetBundleOptions.ForceRebuildAssetBundle,
                    target
                );

                var output = Path.Combine(outDir, bundle.assetBundleName);
                File.Copy(Path.Combine(buildDir, bundle.assetBundleName), output, true);
                Debug.Log($"[BuildShaderBundles] Wrote {output}");
            }
            finally
            {
                PlayerSettings.SetGraphicsAPIs(target, previousApis);
                PlayerSettings.SetUseDefaultGraphicsAPIs(target, useDefaultApis);
            }
        }
    }
}

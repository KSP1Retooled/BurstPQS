using System;
using Unity.Profiling;

namespace BurstPQS.Util;

internal struct SuspendProfileScope : IDisposable
{
    ProfilerMarker.AutoScope scope;

    public SuspendProfileScope(ProfilerMarker.AutoScope scope)
    {
        this.scope = scope;
    }

    public void Dispose()
    {
    }
}

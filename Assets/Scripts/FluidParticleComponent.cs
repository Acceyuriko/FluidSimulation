using System;
using System.Runtime.CompilerServices;
using Unity.Entities;

[GenerateAuthoringComponent]
[Serializable]
public struct FluidParticleComponent : IComponentData, IFormattable
{
    public float radius;
    public float restDensity;
    public float viscosity;
    public float gasConstant;

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public string ToString(string format, IFormatProvider formatProvider)
    {
        return $"FluidParticleComponent(radius: {radius}, restDensity: {restDensity}, viscosity: {viscosity}, gasConstant: {gasConstant})";
    }
}

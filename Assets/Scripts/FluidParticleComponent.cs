using System;
using Unity.Entities;

[GenerateAuthoringComponent]
[Serializable]
public struct FluidParticleComponent : IComponentData
{
    public float density;
    public float viscosity;
}

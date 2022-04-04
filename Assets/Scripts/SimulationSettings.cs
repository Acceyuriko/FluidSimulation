using System;
using Unity.Entities;

[GenerateAuthoringComponent]
[Serializable]
public struct SimulationSettings : IComponentData
{
    public bool UseGPU;
    public float FPS;
}

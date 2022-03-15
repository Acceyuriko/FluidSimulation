using Unity.Entities;
using Unity.Mathematics;

[GenerateAuthoringComponent]
public struct FluidComponent: IComponentData {
    public float radius;
    public float density;
    public float3 gravity;
}
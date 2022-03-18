using Unity.Entities;
using Unity.Mathematics;

[GenerateAuthoringComponent]
public struct FluidComponent: IComponentData {
    public Entity prefab;
}
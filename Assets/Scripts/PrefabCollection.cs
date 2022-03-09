using System;
using Unity.Collections;
using Unity.Entities;
using Unity.Mathematics;

[GenerateAuthoringComponent]
[Serializable]
public struct PrefabCollection : IComponentData
{
    public Entity ParticlePrefab;
}

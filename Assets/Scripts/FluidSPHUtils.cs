using Unity.Burst;
using Unity.Collections;
using Unity.Entities;
using Unity.Jobs;
using Unity.Mathematics;
using Unity.Physics;
using Unity.Physics.Extensions;
using Unity.Transforms;

public static class FluidSPHUtils
{
    public struct SPHData
    {
        public JobHandle dependency;
        public NativeArray<Translation> positions;
        public NativeArray<FluidParticleComponent> particles;
        public NativeArray<PhysicsMass> physicsMasses;
        public NativeArray<PhysicsVelocity> physicsVelocities;
        public NativeArray<Translation> boundaryPositions;
        public NativeArray<BoundaryParticleComponent> boundaryParticles;
        public NativeArray<float> gridSize;
        public NativeArray<float> min;
        public NativeArray<float> max;
        public NativeMultiHashMap<uint, int> grid;
        public NativeMultiHashMap<uint, int> boundaryGrid;
    }

    [BurstCompile]
    private struct FindGridSizeJob : IJobEntityBatch
    {
        [ReadOnly] public ComponentTypeHandle<FluidParticleComponent> particleTypeHandle;
        public NativeArray<float> gridSize;
        public float kernelRadiusRate;

        public void Execute(ArchetypeChunk batchInChunk, int batchIndex)
        {
            var settings = batchInChunk.GetNativeArray(particleTypeHandle);
            var radius = settings[0].radius;
            if (radius < gridSize[0] / kernelRadiusRate)
            {
                gridSize[0] = radius * kernelRadiusRate;
            }
        }
    }

    [BurstCompile]
    private struct FindBoundsJob : IJob
    {
        [ReadOnly] public NativeArray<Translation> positions;
        [ReadOnly] public NativeArray<Translation> boundaryPositions;
        [ReadOnly] public NativeArray<float> gridSize;
        public NativeArray<float> min;
        public NativeArray<float> max;
        public NativeMultiHashMap<uint, int> grid;
        public NativeMultiHashMap<uint, int> boundaryGrid;

        public void Execute()
        {
            for (int i = 0; i < positions.Length; i++)
            {
                if (positions[i].Value.x < min[0]) min[0] = positions[i].Value.x;
                if (positions[i].Value.y < min[1]) min[1] = positions[i].Value.y;
                if (positions[i].Value.z < min[2]) min[2] = positions[i].Value.z;

                if (positions[i].Value.x > max[0]) max[0] = positions[i].Value.x;
                if (positions[i].Value.y > max[1]) max[1] = positions[i].Value.y;
                if (positions[i].Value.z > max[2]) max[2] = positions[i].Value.z;

                grid.Add(Hash(Quantize(positions[i].Value, gridSize[0])), i);
            }

            for (int i = 0; i < boundaryPositions.Length; i++)
            {
                if (boundaryPositions[i].Value.x < min[0]) min[0] = boundaryPositions[i].Value.x;
                if (boundaryPositions[i].Value.y < min[1]) min[1] = boundaryPositions[i].Value.y;
                if (boundaryPositions[i].Value.z < min[2]) min[2] = boundaryPositions[i].Value.z;

                if (boundaryPositions[i].Value.x > max[0]) max[0] = boundaryPositions[i].Value.x;
                if (boundaryPositions[i].Value.y > max[1]) max[1] = boundaryPositions[i].Value.y;
                if (boundaryPositions[i].Value.z > max[2]) max[2] = boundaryPositions[i].Value.z;

                boundaryGrid.Add(Hash(Quantize(boundaryPositions[i].Value, gridSize[0])), i);
            }

            min[0] = math.floor(min[0] / gridSize[0]) * gridSize[0];
            min[1] = math.floor(min[1] / gridSize[0]) * gridSize[0];
            min[2] = math.floor(min[2] / gridSize[0]) * gridSize[0];
            // The grid is divided by left closed and right open,
            // so the maximum value needs to be added with an additional grid
            max[0] = math.ceil(max[0] / gridSize[0]) * gridSize[0] + 1f;
            max[1] = math.ceil(max[1] / gridSize[0]) * gridSize[0] + 1f;
            max[2] = math.ceil(max[2] / gridSize[0]) * gridSize[0] + 1f;
        }
    }

    [BurstCompile]
    public struct ApplyForceJob : IJobEntityBatchWithIndex
    {
        [ReadOnly] public NativeArray<float> densities;
        [ReadOnly] public NativeArray<float3> forces;
        [ReadOnly] public float deltaTime;
        [ReadOnly] public ComponentTypeHandle<PhysicsMass> massTypeHandle;
        public ComponentTypeHandle<PhysicsVelocity> velocityTypeHandle;

        public void Execute(ArchetypeChunk batchInChunk, int batchIndex, int indexOfFirstEntityInQuery)
        {
            var velocities = batchInChunk.GetNativeArray(velocityTypeHandle);
            var masses = batchInChunk.GetNativeArray(massTypeHandle);

            for (int i = 0; i < velocities.Length; i++)
            {
                var globalIndex = i + indexOfFirstEntityInQuery;
                if (densities[globalIndex] == 0)
                {
                    continue;
                }
                var velocity = velocities[i];
                velocity.ApplyLinearImpulse(
                    masses[i],
                    forces[globalIndex] / densities[globalIndex] / masses[i].InverseMass * deltaTime
                );
                velocities[i] = velocity;
            }
        }
    }

    // The kernel function radius is generally 3 ~ 5 times the particle radius
    public static readonly float KernelRadiusRate = 4f;

    public static readonly ComponentType[] ParticleQueryComponentTypes = new ComponentType[] {
        typeof(FluidParticleComponent),
        typeof(Translation),
        typeof(PhysicsVelocity),
        typeof(PhysicsMass),
        typeof(PhysicsCollider),
    };

    public static readonly ComponentType[] BoundaryQueryComponentTypes = new ComponentType[] {
        typeof(BoundaryParticleComponent),
        typeof(Translation)
    };

    public static SPHData InitializeData(
        EntityQuery particleQuery,
        EntityQuery boundaryQuery,
        ComponentTypeHandle<FluidParticleComponent> particleTypeHandle
    )
    {
        var positions = particleQuery.ToComponentDataArrayAsync<Translation>(Allocator.TempJob, out var positionHandle);
        var particles = particleQuery.ToComponentDataArrayAsync<FluidParticleComponent>(Allocator.TempJob, out var particleHandle);
        var physicsMasses = particleQuery.ToComponentDataArrayAsync<PhysicsMass>(Allocator.TempJob, out var physicsMassHandle);
        var physicsVelocities = particleQuery.ToComponentDataArrayAsync<PhysicsVelocity>(Allocator.TempJob, out var physicsVelocityHandle);

        var dependency = JobHandle.CombineDependencies(
            positionHandle,
            particleHandle,
            JobHandle.CombineDependencies(
                physicsMassHandle,
                physicsVelocityHandle
            )
        );

        var boundaryPositions = boundaryQuery.ToComponentDataArrayAsync<Translation>(Allocator.TempJob, out var boundaryPositionHandle);
        var boundaryParticles = boundaryQuery.ToComponentDataArrayAsync<BoundaryParticleComponent>(Allocator.TempJob, out var boundaryParticleHandle);

        dependency = JobHandle.CombineDependencies(
            dependency,
            boundaryPositionHandle,
            boundaryParticleHandle
        );

        var gridSize = new NativeArray<float>(1, Allocator.TempJob);
        gridSize[0] = float.MaxValue;

        var findGridSizeJob = new FindGridSizeJob
        {
            particleTypeHandle = particleTypeHandle,
            gridSize = gridSize,
            kernelRadiusRate = KernelRadiusRate,
        };
        dependency = findGridSizeJob.Schedule(particleQuery, dependency);

        // 计算网格布局
        var min = new NativeArray<float>(3, Allocator.TempJob);
        min[0] = min[1] = min[2] = float.MaxValue;
        var max = new NativeArray<float>(3, Allocator.TempJob);
        max[0] = max[1] = max[2] = float.MinValue;

        var grid = new NativeMultiHashMap<uint, int>(positions.Length, Allocator.TempJob);
        var boundaryGrid = new NativeMultiHashMap<uint, int>(positions.Length, Allocator.TempJob);

        var findBoundsJob = new FindBoundsJob
        {
            positions = positions,
            boundaryPositions = boundaryPositions,
            gridSize = gridSize,
            min = min,
            max = max,
            grid = grid,
            boundaryGrid = boundaryGrid,
        };
        dependency = findBoundsJob.Schedule(dependency);

        return new SPHData
        {
            dependency = dependency,
            positions = positions,
            particles = particles,
            physicsMasses = physicsMasses,
            physicsVelocities = physicsVelocities,
            boundaryPositions = boundaryPositions,
            boundaryParticles = boundaryParticles,
            gridSize = gridSize,
            min = min,
            max = max,
            grid = grid,
            boundaryGrid = boundaryGrid,
        };
    }

    public static void Dispose(SPHData data, JobHandle dependency)
    {
        data.positions.Dispose(dependency);
        data.particles.Dispose(dependency);
        data.physicsMasses.Dispose(dependency);
        data.physicsVelocities.Dispose(dependency);
        data.boundaryPositions.Dispose(dependency);
        data.boundaryParticles.Dispose(dependency);
        data.gridSize.Dispose(dependency);
        data.min.Dispose(dependency);
        data.max.Dispose(dependency);
        data.grid.Dispose(dependency);
        data.boundaryGrid.Dispose(dependency);
    }

    public static int3 Quantize(float3 position, float size)
    {
        return new int3(
            (int)math.floor(position.x / size),
            (int)math.floor(position.y / size),
            (int)math.floor(position.z / size)
        );
    }

    // FNV-1 hash https://en.wikipedia.org/wiki/Fowler%E2%80%93Noll%E2%80%93Vo_hash_function 
    public static uint Hash(int3 p)
    {
        uint hash = 2166136261u;

        for (int i = 0; i < 3; i++)
        {
            for (int j = 0; j < sizeof(int); j++)
            {
                byte b = (byte)(p[i] >> (j * 8));
                hash *= 16777619u;
                hash ^= b;
            }
        }

        return hash;
    }
}
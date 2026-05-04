from coppeliasim_zmqremoteapi_client import RemoteAPIClient

client = RemoteAPIClient()
sim = client.getObject('sim')

client.setStepping(False)
sim.startSimulation()

print("Connected to CoppeliaSim and simulation started!")

print("\n=== Available joints in scene ===")

index = 0
while True:
    obj = sim.getObjects(index, sim.object_joint_type)
    
    if obj == -1:
        break

    print(sim.getObjectAlias(obj, 1))
    index += 1

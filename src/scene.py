# scene_builder.py
import genesis as gs

def build_orca_scene(show_viewer: bool = False):
    """
    Initializes the Genesis engine and constructs the physical world simulation.
    
    Args:
        show_viewer (bool): If True, opens a 3D GUI window for visual debugging.
                            Set to False for headless LeRobot training.
                            
    Returns:
        tuple: (gs.Scene, dict) containing the compiled scene object and a 
               dictionary of references to all instantiated entities.
    """
    try:
        gs.init(backend=gs.gpu)
    except gs.GenesisException:
        # Catch exception if the engine is accidentally re-initialized within the same process
        pass
        
    scene = gs.Scene(
        # sim_options=gs.options.SimOptions(dt=0.01, gravity=(0, 0, -9.81)),
        viewer_options=gs.options.ViewerOptions(
            camera_pos=(1, 1, 2),
            camera_lookat=(0, 0, 1),
            res=(1280, 720),
            max_FPS=60,
        ),
        show_viewer=show_viewer
    )
    
    # 1. Environment Ground Plane
    plane = scene.add_entity(gs.morphs.Plane())
    
    # 2. Red Sphere (The target object that needs to be pushed/manipulated)
    sphere = scene.add_entity(
        morph=gs.morphs.Sphere(pos=(0.0, 0.5, 0.1), radius=0.08),
        surface=gs.surfaces.Default(color=(1.0, 0.0, 0.0))
    )
    
    # 3. Green Box
    box = scene.add_entity(
        morph=gs.morphs.Box(pos=(0.5, 0.5, 0.05), size=(0.3, 0.3, 0.1), fixed=True),
        surface=gs.surfaces.Default(color=(0.0, 1.0, 0.0)),
    )
    
    # 4. Blue Cube
    cube = scene.add_entity(
        morph=gs.morphs.Box(pos=(0.0, 0.0, 0.1), size=(0.1, 0.1, 0.1)),
        surface=gs.surfaces.Default(color=(0.0, 0.0, 1.0))
    )
    
    # 5. Virtual Camera
    camera = scene.add_camera(
        res=(640, 480),
        pos=(1.5, 1.5, 1.2),
        lookat=(0.3, 0.3, 0.1),
        fov=45
    )
    
    scene.build()
    
    entities = {
        "plane": plane,
        "sphere": sphere,
        "box": box,
        "cube": cube,
        "camera": camera
    }
    
    return scene, entities
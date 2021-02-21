"""
The template of the main script of the machine learning process
"""

import games.arkanoid.communication as comm
from games.arkanoid.communication import ( \
    SceneInfo, GameStatus, PlatformAction
)

def ml_loop():
    """
    The main loop of the machine learning process

    This loop is run in a separate process, and communicates with the game process.

    Note that the game process won't wait for the ml process to generate the
    GameInstruction. It is possible that the frame of the GameInstruction
    is behind of the current frame in the game process. Try to decrease the fps
    to avoid this situation.
    """

    # === Here is the execution order of the loop === #
    # 1. Put the initialization code here.
    ball_served = False
    pre_Ball_x=95
    pre_Ball_y=400
    m=0
    # 2. Inform the game process that ml process is ready before start the loop.
    comm.ml_ready()

    # 3. Start an endless loop.
    while True:
        # 3.1. Receive the scene information sent from the game process.
        scene_info = comm.get_scene_info()

        # 3.2. If the game is over or passed, the game process will reset
        #      the scene and wait for ml process doing resetting job.
        if scene_info.status == GameStatus.GAME_OVER or \
            scene_info.status == GameStatus.GAME_PASS:
            # Do some stuff if needed
            ball_served = False
            pre_Ball_x=scene_info.ball[0]
            pre_ball_y=scene_info.ball[1]
            print(pre_Ball_x,pre_Ball_y)
            # 3.2.1. Inform the game process that ml process is ready
            comm.ml_ready()
            continue

        # 3.3. Put the code here to handle the scene information
        
        ball_x=scene_info.ball[0]
        ball_y=scene_info.ball[1]
        platform_x=scene_info.platform[0]
        Vx=ball_x-pre_Ball_x
        Vy=ball_y-pre_Ball_y
        
        # 3.4. Send the instruction for this frame to the game process
        if not ball_served:
            #comm.send_instruction(scene_info.frame, PlatformAction.MOVE_LEFT)
            #comm.send_instruction(scene_info.frame, PlatformAction.MOVE_LEFT)
            comm.send_instruction(scene_info.frame, PlatformAction.SERVE_TO_RIGHT)
            ball_served = True
            if Vy>0:
                newp=down(ball_x,ball_y,Vx,scene_info)
                if platform_x+10>newp:
                    comm.send_instruction(scene_info.frame, PlatformAction.MOVE_LEFT)
                elif platform_x+30<newp:
                    comm.send_instruction(scene_info.frame, PlatformAction.MOVE_RIGHT)
            
            else:
                newp=up(ball_x,ball_y,Vx,scene_info)
                if platform_x+10>newp:
                    comm.send_instruction(scene_info.frame, PlatformAction.MOVE_LEFT)
                elif platform_x+30<newp:
                    comm.send_instruction(scene_info.frame, PlatformAction.MOVE_RIGHT)
            
            pre_Ball_x=ball_x
            pre_Ball_y=ball_y
def down(x,y,Vx,scene_info):
    while 1:
        if y>400:
            break
        if Vx>=0:
            x+=7
            y+=7
            if x>=193:
                x=400-x
                Vx=-Vx
                continue
            for br in scene_info.bricks:
                if x>br[0] and x<br[0]+25 and y<br[1]+10 and y>br[1]-5:
                    if y-7<br[1]:
                        return up(x,y,Vx,scene_info)
                    else:
                        Vx=-Vx
                    break
        else:
            x-=7
            y+=7
            if x<=0:
                x=-x
                Vx=-Vx
                continue
            for br in scene_info.bricks:
                if x>br[0] and x<br[0]+25 and y<br[1]+10 and y>br[1]-5:
                    if y-7<br[1]:
                        return up(x,y,Vx,scene_info)
                    else:
                        Vx=-Vx
                    break
    return x

def up(x,y,Vx,scene_info):
    while 1:
        if y<0:
            break
        if Vx>0:
            x+=7
            y-=7
            if x>=193:
                x=400-x
                Vx=-Vx
                continue
            for br in scene_info.bricks:
                   if x>br[0] and x<br[0]+25 and y<br[1]+10 and y>br[1]-5:
                       if y+7>br[1]+10: #hit bitton
                           return down(x,y,Vx,scene_info)
                       else:
                           Vx=-Vx
                           break
        else:
            x-=7
            y-=7
            if x<=0:
                x=-x
                Vx=-Vx
                continue
            for br in scene_info.bricks:
                   if x>br[0] and x<br[0]+25 and y<br[1]+10 and y>br[1]-5:
                       if y+7>br[1]+10:
                           return down(x,y,Vx,scene_info)
                       else:
                           Vx=-Vx
                           break
    return x               

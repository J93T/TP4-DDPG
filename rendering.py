import sys
import select

#Rendering can be switched on/off by pressing Enter

def rendering(env,render,r):

    i, o, e = select.select([sys.stdin], [], [], 0.0001)
    for s in i:
        if s == sys.stdin:
            inputs = sys.stdin.readline()
            if render:
                render = False
            else:
                render = True
    if render:
        env.render()
        #print(r)
    return render

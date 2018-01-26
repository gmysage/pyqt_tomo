#from matplotlib import pyplot as plt
#
#class LineBuilder:
#    def __init__(self, line):
#        self.line = line
#        self.xs = list(line.get_xdata())
#        self.ys = list(line.get_ydata())
#        self.cid = line.figure.canvas.mpl_connect('button_press_event', self)
#
#    def __call__(self, event):
#        print('click', event)
#        if event.inaxes!=self.line.axes: return
#        self.xs.append(event.xdata)
#        self.ys.append(event.ydata)
#        self.line.set_data(self.xs, self.ys)
#        self.line.figure.canvas.draw()
#
#fig = plt.figure()
#ax = fig.add_subplot(111)
#ax.set_title('click to build line segments')
#line, = ax.plot([0], [0])  # empty line
#linebuilder = LineBuilder(line)
#
#plt.show()

####################################################################
#import numpy as np
#import matplotlib.pyplot as plt
#
#class DraggableRectangle:
#    def __init__(self, rect):
#        self.rect = rect
#        self.press = None
#
#    def connect(self):
#        'connect to all the events we need'
#        self.cidpress = self.rect.figure.canvas.mpl_connect(
#            'button_press_event', self.on_press)
#        self.cidrelease = self.rect.figure.canvas.mpl_connect(
#            'button_release_event', self.on_release)
#        self.cidmotion = self.rect.figure.canvas.mpl_connect(
#            'motion_notify_event', self.on_motion)
#
#    def on_press(self, event):
#        'on button press we will see if the mouse is over us and store some data'
#        if event.inaxes != self.rect.axes: return
#
#        contains, attrd = self.rect.contains(event)
#        if not contains: return
#        print('event contains', self.rect.xy)
#        x0, y0 = self.rect.xy
#        self.press = x0, y0, event.xdata, event.ydata
#
#    def on_motion(self, event):
#        'on motion we will move the rect if the mouse is over us'
#        if self.press is None: return
#        if event.inaxes != self.rect.axes: return
#        x0, y0, xpress, ypress = self.press
#        dx = event.xdata - xpress
#        dy = event.ydata - ypress
#        #print('x0=%f, xpress=%f, event.xdata=%f, dx=%f, x0+dx=%f' %
#        #      (x0, xpress, event.xdata, dx, x0+dx))
#        self.rect.set_x(x0+dx)
#        self.rect.set_y(y0+dy)
#
#        self.rect.figure.canvas.draw()
#
#
#    def on_release(self, event):
#        'on release we reset the press data'
#        self.press = None
#        self.rect.figure.canvas.draw()
#
#    def disconnect(self):
#        'disconnect all the stored connection ids'
#        self.rect.figure.canvas.mpl_disconnect(self.cidpress)
#        self.rect.figure.canvas.mpl_disconnect(self.cidrelease)
#        self.rect.figure.canvas.mpl_disconnect(self.cidmotion)
#
#fig = plt.figure()
#ax = fig.add_subplot(111)
#rects = ax.bar(range(10), 20*np.random.rand(10))
#drs = []
#for rect in rects:
#    dr = DraggableRectangle(rect)
#    dr.connect()
#    drs.append(dr)
#
#plt.show()

###################################################################


#        print('hello')
#        self.draw()
import numpy as np
import matplotlib.pylab as plt 

from matplotlib.widgets import RectangleSelector
def line_select_callback(eclick, erelease):
    x1, y1 = eclick.xdata, eclick.ydata
    x2, y2 = erelease.xdata, erelease.ydata
    rect = plt.Rectangle( (min(x1,x2),min(y1,y2)), np.abs(x1-x2), np.abs(y1-y2) )
    axes.add_patch(rect)
    print('efd')
        
        
fig = plt.figure()
axes = fig.add_subplot(111)
axes.plot([1,2],[3,4])

tg = RectangleSelector(axes, line_select_callback,
                       drawtype='box', useblit=False, button=[1,3], 
                       minspanx=5, minspany=5, spancoords='pixels', 
                       interactive=True)   
plt.connect('key_press_envent', tg) 
plt.show()        
####        
        
        
        
        
        
        
        
        
        
        

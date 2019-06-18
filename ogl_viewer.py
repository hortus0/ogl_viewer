import glfw
import numpy as np
from OpenGL.GL import *
from OpenGL.GLU import *
from OpenGL.GLUT import *
from OpenGL.arrays import vbo


class Scene:
    """ OpenGL 2D scene class """

    # initialization
    def __init__(self, width, height, file):
        # time
        self.t = 0
        self.show_object = True
        self.show_shadow = True
        self.perspective_projection = False
        self.width = width
        self.height = height
        self.obj = Object(file)
        self.point_size = 3
        glPointSize(self.point_size)
        glLineWidth(self.point_size)

        # set height of camera
        self.camera_height = 2

        # light position
        self.l0_pos = [10, 20, 0]

        # create VBOs
        self.plane_vbo = vbo.VBO(np.array([
            [-100, 0, 100],
            [100, 0, 100],
            [100, 0, -100],
            [-100, 0, 100],
            [100, 0, -100],
            [-100, 0, -100]
        ]))

        # self.index_vbo = vbo.VBO(np.array(self.scene.obj.v_indices, 'uint'))
        self.vector_vbo = vbo.VBO(np.array(self.obj.sorted_vectors, 'f'))
        self.normal_vbo = vbo.VBO(np.array(self.obj.sorted_normals, 'f'))

        # shadow matrix
        self.shadow_mat = [1.0, 0, 0, 0, 0, 1.0, 0, -1.0 / self.l0_pos[1], 0, 0, 1.0, 0, 0, 0, 0, 0]

        # values for rotation
        self.start_p = []
        self.move_p = 0
        self.end_p = 0
        self.act_ori = 1
        self.prevX = 0
        self.prevY = 0
        self.angle = 0
        self.axis = [0, 1, 0]

    # render
    def render(self, shadow):
        if shadow:
            glMaterialfv(GL_FRONT, GL_AMBIENT_AND_DIFFUSE, (0, 0, 0))
        else:
            color = self.obj.color
            glMaterialfv(GL_FRONT, GL_AMBIENT_AND_DIFFUSE, (color[0], color[1], color[2]))

        glPolygonMode(GL_FRONT_AND_BACK, GL_FILL)

        # render scene
        self.vector_vbo.bind()
        glVertexPointerf(self.vector_vbo)
        self.vector_vbo.unbind()

        self.normal_vbo.bind()
        glNormalPointer(GL_FLOAT, 0, self.normal_vbo)
        self.normal_vbo.unbind()

        glEnableClientState(GL_VERTEX_ARRAY)
        glEnableClientState(GL_NORMAL_ARRAY)

        glDrawArrays(GL_TRIANGLES, 0, len(self.obj.sorted_vectors))

        glDisableClientState(GL_NORMAL_ARRAY)
        glDisableClientState(GL_VERTEX_ARRAY)


class Object:
    def __init__(self, file):
        self.file = file
        self.height = 0
        self.width = 0
        self.polygons = []
        self.vectors = []
        self.textures = []
        self.normals = []
        self.vertex_normals = {}
        self.obj_origin = []
        self.bounding_box = []
        self.scale = 0
        self.v_indices = []
        self.sorted_normals = []
        self.sorted_vectors = []
        self.color = [0.8, 0.8, 0.8]
        self.delta_y = 0
        self.import_data()

    def import_data(self):
        with open(self.file) as file:
            for line in file:
                line = line.strip().split()
                if not line:
                    continue
                if line[0] == 'f':
                    vertices = []
                    for v in line[1:]:
                        if '//' in v:
                            temp = v.split('//')
                            vertices.append([int(temp[0]) - 1, '', int(temp[1]) - 1])
                        elif '/' in v:
                            temp = v.split('/')
                            vertices.append([int(temp[0]) - 1, int(temp[1]) - 1, int(temp[2]) - 1])
                        else:
                            vertices.append([int(v) - 1, '', ''])

                    self.polygons.append(vertices)

                if line[0] == 'v':
                    v = np.array([float(line[1]), float(line[2]), float(line[3])])
                    self.vectors.append(v)
                elif line[0] == 'vt':
                    vt = np.array([float(line[1]), float(line[2]), float(line[3])])
                    self.textures.append(vt)
                elif line[0] == 'vn':
                    vn = np.array([float(line[1]), float(line[2]), float(line[3])])
                    self.normals.append(vn)

        self.compute_bounding_box()
        self.compute_scale()
        self.move_up()

        # filling vector array
        if self.polygons[0][0][2] != "":
            for p in self.polygons:
                for vertex in p:
                    self.sorted_vectors.append(self.vectors[vertex[0]])
                    self.sorted_normals.append(self.normals[vertex[2]])

        else:
            for p in self.polygons:
                polygon_points = []
                i = 0
                for vertex in p:
                    self.sorted_vectors.append(self.vectors[vertex[0]])
                    #self.sorted_normals.append([0, 1, -1])
                    polygon_points.append(self.vectors[vertex[0]])

                norm = self.calc_normals(polygon_points[0], polygon_points[1], polygon_points[2])
                # norm = norm * (-1)
                self.sorted_normals.append(norm)
                self.sorted_normals.append(norm)
                self.sorted_normals.append(norm)

    def compute_bounding_box(self):
        temp = [list(map(min, zip(*self.vectors))), list(map(max, zip(*self.vectors)))]
        self.bounding_box = (np.array([temp[0][0], temp[0][1], temp[0][2]]), np.array([temp[1][0], temp[1][1], temp[1][2]]))
        self.height = self.bounding_box[1][1] - self.bounding_box[0][1]
        self.width = self.bounding_box[1][0] - self.bounding_box[0][0]
        min_array = self.bounding_box[0]
        max_array = self.bounding_box[1]

        mid_x = min_array[0] + ((max_array[0] - min_array[0]) / 2)
        mid_y = min_array[1] + ((max_array[1] - min_array[1]) / 2)
        mid_z = min_array[2] + ((max_array[2] - min_array[2]) / 2)

        self.obj_origin = np.array([mid_x, mid_y, mid_z])

        self.vectors = self.vectors - self.obj_origin

        self.obj_origin = [0, 0, 0]

    def compute_scale(self):
        # calculate scale
        scaleX = abs((self.bounding_box[1][0] - self.bounding_box[0][0]))
        scaleY = abs((self.bounding_box[1][1] - self.bounding_box[0][1]))
        scaleZ = abs((self.bounding_box[1][2] - self.bounding_box[0][2]))

        self.scale = 2 / max(scaleX, scaleY, scaleZ)

        for v in self.vectors:
            v *= self.scale

    def move_up(self):
        min_y = min([x[1] for x in self.vectors])
        max_y = max([x[1] for x in self.vectors])
        self.delta_y = (max_y - min_y) / 2
        for v in self.vectors:
            v[1] = v[1] + self.delta_y
        self.obj_origin[1] += self.delta_y

    @staticmethod
    def normalize(x):
        return x / np.linalg.norm(x)

    @staticmethod
    def calc_normals(x, y, z):
        xy = Object.normalize(y) - Object.normalize(x)
        xz = Object.normalize(z) - Object.normalize(x)
        n = np.cross(xy, xz)
        return n

class RenderWindow:
    """GLFW Rendering window class"""

    def __init__(self, file):

        # save current working directory
        cwd = os.getcwd()

        # Initialize the library
        if not glfw.init():
            return

        # restore cwd
        os.chdir(cwd)

        # buffer hints
        glfw.window_hint(glfw.DEPTH_BITS, 32)

        # define desired frame rate
        self.frame_rate = 100

        # make a window
        self.width, self.height = 800, 800
        self.aspect = self.width / float(self.height)
        self.window = glfw.create_window(self.width, self.height, "2D Graphics", None, None)
        if not self.window:
            glfw.terminate()
            return

        # Make the window's context current
        glfw.make_context_current(self.window)

        # create 3D
        self.scene = Scene(self.width, self.height, file)

        # initialize GL
        glViewport(0, 0, self.width, self.height)
        glEnable(GL_DEPTH_TEST)
        glEnable(GL_NORMALIZE)
        glEnable(GL_LIGHTING)
        glEnable(GL_LIGHT0)
        glLightfv(GL_LIGHT0, GL_POSITION, self.scene.l0_pos)
        #glShadeModel(GL_FLAT)

        glClearColor(1.0, 1.0, 0.5, 0.0)
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        glOrtho(-1, 1, -1, 1, -100, 100)
        glMatrixMode(GL_MODELVIEW)

        gluLookAt(0, 1, 3, 0, self.scene.obj.delta_y, 0, 0, 1, 0)

        # set window callbacks
        glfw.set_mouse_button_callback(self.window, self.on_mouse_button)
        glfw.set_scroll_callback(self.window, self.on_scroll)
        glfw.set_cursor_pos_callback(self.window, self.mouse_moved)
        glfw.set_key_callback(self.window, self.on_keyboard)
        glfw.set_window_size_callback(self.window, self.on_size)

        self.rotation_speed = 50
        self.translation_speed = 20
        self.zoom_speed = 100

        # exit flag
        self.exitNow = False

        # animation flag
        self.animation = False

        # rotation flag
        self.do_rotation = False

        # translation flag
        self.do_translate = False

        # zoom flag
        self.do_zoom = False

        # shift flag
        self.left_shift_pressed = False

    def set_camera(self):
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        if not self.scene.perspective_projection:
            glOrtho(-1, 1, -1, 1, -100, 100)
        else:
            gluPerspective(45, self.aspect, 0.1, 100)
        glMatrixMode(GL_MODELVIEW)

    def project_on_sphere(self, x, y, r):
        x, y = x-self.width / 2.0, self.height / 2.0-y
        a = min(r*r, x**2 + y**2)
        z = np.abs(np.sqrt(r*r - a))
        l = np.sqrt(x**2 + y**2 + z**2)
        return x / l, y / l, z / l

    def on_mouse_button(self, win, button, action, mods):
        if button == glfw.MOUSE_BUTTON_LEFT:
            r = min(self.width, self.height) / 2.0
            if action == glfw.PRESS:
                self.do_rotation = True
                x, y = glfw.get_cursor_pos(win)
                self.scene.start_p = self.project_on_sphere(x, y, r)
            elif action == glfw.RELEASE:
                self.do_rotation = False
                self.scene.angle = 0

        elif button == glfw.MOUSE_BUTTON_RIGHT:
            if action == glfw.PRESS:
                self.do_translate = True
                self.scene.start_p = glfw.get_cursor_pos(win)
            elif action == glfw.RELEASE:
                self.do_translate = False

        elif button == glfw.MOUSE_BUTTON_MIDDLE:
            if action == glfw.PRESS:
                self.do_zoom = True
                self.scene.start_p = 0
            elif action == glfw.RELEASE:
                self.do_zoom = False

    def mapToRange(self, x, fromRange, toRange):
        r = (toRange[1] - toRange[0]) / (fromRange[1] - fromRange[0])
        return (x - fromRange[0]) * r + toRange[0]

    def mouse_moved(self, win, x, y):
        if self.do_rotation:
            r = min(self.width, self.height) / 2.0

            move_p = self.project_on_sphere(x, y, r)
            if np.dot(self.scene.start_p, move_p) <= 1 and np.dot(self.scene.start_p, move_p) >= 0:
                self.scene.angle = np.arccos(np.dot(self.scene.start_p, move_p))
                self.scene.axis = np.cross(self.scene.start_p, move_p)

                o = self.scene.obj.obj_origin
                glTranslatef(o[0], o[1], o[2])
                glRotatef(self.rotation_speed * self.scene.angle, self.scene.axis[0], self.scene.axis[1], self.scene.axis[2])
                glTranslatef(-o[0], -o[1], -o[2])

                self.scene.start_p = self.project_on_sphere(x, y, r)

        elif self.do_translate:
            move_x, move_y = self.scene.start_p[0] - x, self.scene.start_p[1] - y
            x = self.mapToRange(move_x, (0, self.width), (0, 1))
            y = self.mapToRange(move_y, (0, self.height), (0, 1))

            glTranslatef(self.translation_speed * (-x), self.translation_speed * y, 0)
            self.scene.start_p = self.scene.start_p[0] - move_x, self.scene.start_p[1] - move_y

    def on_scroll(self, win, x_offset, y_offset):
        scale = 1 + y_offset / self.zoom_speed
        glScalef(scale, scale, scale)

    def on_keyboard(self, win, key, scancode, action, mods):
        if key == glfw.KEY_LEFT_SHIFT:
            if action == glfw.PRESS:
                self.left_shift_pressed = True
            elif action == glfw.RELEASE:
                self.left_shift_pressed = False

        if action == glfw.PRESS or action == glfw.REPEAT:
            # Q to quit
            if key == glfw.KEY_Q:
                self.exitNow = True

            # rotate with keys X, Y, Z
            o = self.scene.obj.obj_origin
            if key == glfw.KEY_Z:
                glTranslatef(o[0], o[1], o[2])
                glRotatef(10, 0, 1.0, 0)
                glTranslatef(-o[0], -o[1], -o[2])
            if key == glfw.KEY_X:
                glTranslatef(o[0], o[1], o[2])
                glRotatef(10, 1.0, 0, 0)
                glTranslatef(-o[0], -o[1], -o[2])
            if key == glfw.KEY_Y:
                glTranslatef(o[0], o[1], o[2])
                glRotatef(10, 0, 0, 1.0)
                glTranslatef(-o[0], -o[1], -o[2])

            # change projection mode
            if key == glfw.KEY_O:
                self.scene.perspective_projection = False
                self.set_camera()
            if key == glfw.KEY_P:
                self.scene.perspective_projection = True
                self.set_camera()

            # change visibility of object
            if key == glfw.KEY_D:
                self.scene.show_object = not self.scene.show_object

            # change color
            if key == glfw.KEY_S:
                if self.left_shift_pressed:
                    glClearColor(0, 0, 0, 0)
                else:
                    self.scene.obj.color = [0, 0, 0]

            if key == glfw.KEY_W:
                if self.left_shift_pressed:
                    glClearColor(1, 1, 1, 0)
                else:
                    self.scene.obj.color = [1, 1, 1]

            if key == glfw.KEY_R:
                if self.left_shift_pressed:
                    glClearColor(1, 0, 0, 0)
                else:
                    self.scene.obj.color = [1, 0, 0]

            if key == glfw.KEY_G:
                if self.left_shift_pressed:
                    glClearColor(0, 1, 0, 0)
                else:
                    self.scene.obj.color = [0, 1, 0]

            if key == glfw.KEY_B:
                if self.left_shift_pressed:
                    glClearColor(0, 0, 1, 0)
                else:
                    self.scene.obj.color = [0, 0, 1]

            # enable / disable shadow
            if key == glfw.KEY_H:
                self.scene.show_shadow = not self.scene.show_shadow

            # reset object
            if key == glfw.KEY_1:
                self.scene.obj.color = [0.8, 0.8, 0.8]
                glClearColor(1, 1, 0.5, 0)
                self.set_camera()
                self.scene.camera_height = 3
                glLoadIdentity()

                gluLookAt(0, 1, 3, 0, 0, 0, 0, 1, 0)

                o = self.scene.obj.obj_origin
                glTranslatef(-o[0], -o[1], -o[2])

            if key == glfw.KEY_LEFT:
                self.scene.l0_pos[0] -= 1
                glLightfv(GL_LIGHT0, GL_POSITION, self.scene.l0_pos)
            if key == glfw.KEY_RIGHT:
                self.scene.l0_pos[0] += 1
                glLightfv(GL_LIGHT0, GL_POSITION, self.scene.l0_pos)

    def on_size(self, win, width, height):
        self.width = width
        self.height = height
        self.aspect = width / float(height)
        glViewport(0, 0, width, height)
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()

        if self.scene.perspective_projection:
            if width <= height:
                gluPerspective(45, float(width) / height, 0.1, 100)
            else:
                gluPerspective(45, float(height) / width, 0.1, 100)
        else:
            if width <= height:
                glOrtho(-1, 1,
                        -1 * height / width, 1 * height / width,
                        -10.0, 10.0)
            else:
                glOrtho(-1 * width / height, 1 * width / height,
                        -1, 1,
                        -10.0, 10.0)

        glMatrixMode(GL_MODELVIEW)

    def draw_coordinate_system(self):
        glPushMatrix()
        glLoadIdentity()
        # x-axis
        glColor3f(1.0, 0.0, 0.0)  # red
        glBegin(GL_LINES)
        glVertex3f(-4.0, 0.0, 0.0)
        glVertex3f(4.0, 0.0, 0.0)
        glEnd()

        # y-axis
        glColor3f(0.0, 1.0, 0.0)  # green
        glBegin(GL_LINES)
        glVertex3f(0.0, -4.0, 0.0)
        glVertex3f(0.0, 4.0, 0.0)
        glEnd()

        # z-axis
        glColor3f(0.0, 0.0, 1.0)  # blue
        glBegin(GL_LINES)
        glVertex3f(0.0, 0.0, -4.0)
        glVertex3f(0.0, 0.0, 4.0)
        glEnd()
        glPopMatrix()

    def run(self):
        # initializer timer
        glfw.set_time(0.0)
        t = 0.0
        while not glfw.window_should_close(self.window) and not self.exitNow:
            # update every x seconds
            curr_t = glfw.get_time()
            if curr_t - t > 1.0 / self.frame_rate:
                # update time
                t = curr_t

                # clear
                glEnable(GL_DEPTH_TEST)
                glEnable(GL_LIGHTING)
                glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

                if self.scene.show_object:
                    self.scene.render(False)

                if self.scene.show_shadow:
                    # drawing shadow
                    glPushMatrix()
                    glTranslatef(self.scene.l0_pos[0], self.scene.l0_pos[1], self.scene.l0_pos[2])
                    glMultMatrixf(self.scene.shadow_mat)
                    glTranslatef(-self.scene.l0_pos[0], -self.scene.l0_pos[1], -self.scene.l0_pos[2])

                    # render object again
                    self.scene.render(True)

                    glPopMatrix()

                glDisable(GL_LIGHTING)
                self.draw_coordinate_system()

                glfw.swap_buffers(self.window)
                # Poll for and process events
                glfw.poll_events()
        # end
        glfw.terminate()


if __name__ == '__main__':
    if len(sys.argv) == 1:
        print("No filepath to obj-file")
        sys.exit(-1)

    print("Rendering ", sys.argv[1])
    rw = RenderWindow(sys.argv[1])
    rw.run()

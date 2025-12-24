import math
import pygame
import sys

# =================================================================================
# BAGIAN 1: SISTEM MATEMATIKA MANUAL (Berdasarkan Bab II Teori)
# =================================================================================

class Vector3:
    def __init__(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z

    def __repr__(self):
        return f"Vec3({self.x:.2f}, {self.y:.2f}, {self.z:.2f})"

    # Implementasi Rumus 1: Penjumlahan Vektor
    def add(self, other):
        return Vector3(self.x + other.x, self.y + other.y, self.z + other.z)

    # Implementasi Rumus: Pengurangan Vektor
    def sub(self, other):
        return Vector3(self.x - other.x, self.y - other.y, self.z - other.z)

    # Implementasi Rumus 2: Perkalian Skalar
    def mul(self, scalar):
        return Vector3(self.x * scalar, self.y * scalar, self.z * scalar)

    # Implementasi Rumus 3: Dot Product
    def dot(self, other):
        return self.x * other.x + self.y * other.y + self.z * other.z
    
    # Implementasi Rumus: Cross Product
    def cross(self, other):
        return Vector3(
            self.y * other.z - self.z * other.y,
            self.z * other.x - self.x * other.z,
            self.x * other.y - self.y * other.x
        )

    # Implementasi: Normalisasi Vektor
    def magnitude(self):
        return math.sqrt(self.x**2 + self.y**2 + self.z**2)

    def normalize(self):
        mag = self.magnitude()
        if mag == 0:
            return Vector3(0, 0, 0)
        return Vector3(self.x / mag, self.y / mag, self.z / mag)

class Matrix4:
    def __init__(self, data=None):
        # Matriks 4x4 direpresentasikan sebagai List of Lists
        # Baris x Kolom
        if data:
            self.m = data
        else:
            self.m = [[0]*4 for _ in range(4)]

    # Implementasi: Matriks Identitas
    @staticmethod
    def identity():
        mat = Matrix4()
        for i in range(4):
            mat.m[i][i] = 1.0
        return mat

    # Implementasi: Perkalian Matriks 4x4 dengan Matriks 4x4
    def multiply(self, other):
        result = Matrix4()
        for i in range(4): # Baris result
            for j in range(4): # Kolom result
                sum_val = 0
                for k in range(4):
                    sum_val += self.m[i][k] * other.m[k][j]
                result.m[i][j] = sum_val
        return result

    # Implementasi: Perkalian Matriks dengan Vektor (Transformasi Titik)
    # Asumsi vektor input adalah (x, y, z, 1) untuk posisi
    def transform_point(self, vec):
        x = vec.x
        y = vec.y
        z = vec.z
        w = 1.0

        res_x = self.m[0][0]*x + self.m[0][1]*y + self.m[0][2]*z + self.m[0][3]*w
        res_y = self.m[1][0]*x + self.m[1][1]*y + self.m[1][2]*z + self.m[1][3]*w
        res_z = self.m[2][0]*x + self.m[2][1]*y + self.m[2][2]*z + self.m[2][3]*w
        res_w = self.m[3][0]*x + self.m[3][1]*y + self.m[3][2]*z + self.m[3][3]*w

        # Jika w bukan 1 (misal proyeksi), kita idealnya membagi dengan w (perspective divide),
        # tapi untuk transformasi affine biasa w akan tetap 1.
        return Vector3(res_x, res_y, res_z)

# =================================================================================
# BAGIAN 2: TRANSFORMASI GEOMETRI (Berdasarkan Bab II.D)
# =================================================================================

def get_translation_matrix(tx, ty, tz):
    # Implementasi Matriks Translasi (Persamaan 5 / Gambar 2)
    mat = Matrix4.identity()
    mat.m[0][3] = tx
    mat.m[1][3] = ty
    mat.m[2][3] = tz
    return mat

def get_scaling_matrix(sx, sy, sz):
    # Implementasi Matriks Penskalaan (Gambar 3)
    mat = Matrix4.identity()
    mat.m[0][0] = sx
    mat.m[1][1] = sy
    mat.m[2][2] = sz
    return mat

def get_rotation_y_matrix(angle_rad):
    # Implementasi Matriks Rotasi sumbu Y
    c = math.cos(angle_rad)
    s = math.sin(angle_rad)
    mat = Matrix4.identity()
    mat.m[0][0] = c
    mat.m[0][2] = s
    mat.m[2][0] = -s
    mat.m[2][2] = c
    return mat

def get_rotation_x_matrix(angle_rad):
    # Implementasi Matriks Rotasi sumbu X
    c = math.cos(angle_rad)
    s = math.sin(angle_rad)
    mat = Matrix4.identity()
    mat.m[1][1] = c
    mat.m[1][2] = -s
    mat.m[2][1] = s
    mat.m[2][2] = c
    return mat

# =================================================================================

# =================================================================================
# BAGIAN 2.5: KAMERA & RASTERISASI (DEPTH MAP)
# =================================================================================

def get_look_at_matrix(eye, target, up):
    """
    Implementasi View Matrix (LookAt).
    Mengubah koordinat dunia ke koordinat kamera (atau cahaya).
    """
    # Z axis: eye - target (reversed because camera looks down -Z)
    z_axis = Vector3(eye.x - target.x, eye.y - target.y, eye.z - target.z).normalize()
    
    # X axis: cross(up, z)
    x_axis = Vector3(up.y * z_axis.z - up.z * z_axis.y,
                     up.z * z_axis.x - up.x * z_axis.z,
                     up.x * z_axis.y - up.y * z_axis.x).normalize() 
    
    # Y axis: cross(z, x)
    y_axis = Vector3(z_axis.y * x_axis.z - z_axis.z * x_axis.y,
                     z_axis.z * x_axis.x - z_axis.x * x_axis.z,
                     z_axis.x * x_axis.y - z_axis.y * x_axis.x)

    # Matriks Orientasi
    orientation = Matrix4()
    orientation.m[0][0] = x_axis.x; orientation.m[0][1] = x_axis.y; orientation.m[0][2] = x_axis.z
    orientation.m[1][0] = y_axis.x; orientation.m[1][1] = y_axis.y; orientation.m[1][2] = y_axis.z
    orientation.m[2][0] = z_axis.x; orientation.m[2][1] = z_axis.y; orientation.m[2][2] = z_axis.z
    
    # Matriks Translasi Negatif
    translation = get_translation_matrix(-eye.x, -eye.y, -eye.z)
    
    return orientation.multiply(translation)

def get_ortho_matrix(left, right, bottom, top, near, far):
    """
    Implementasi Matriks Proyeksi Orthographic.
    """
    mat = Matrix4.identity()
    mat.m[0][0] = 2.0 / (right - left)
    mat.m[1][1] = 2.0 / (top - bottom)
    mat.m[2][2] = -2.0 / (far - near)
    mat.m[0][3] = -(right + left) / (right - left)
    mat.m[1][3] = -(top + bottom) / (top - bottom)
    mat.m[2][3] = -(far + near) / (far - near)
    return mat

class DepthBuffer:
    """
    Kelas untuk menyimpan Depth Map (Peta Kedalaman).
    Hanya menyimpan nilai Z (float).
    """
    def __init__(self, width, height):
        self.width = width
        self.height = height
        self.buffer = [1.0] * (width * height)
    
    def clear(self):
        # Reset ke kedalaman terjauh (1.0 dalam clip space)
        self.buffer = [1.0] * (self.width * self.height)

    def set_depth(self, x, y, depth):
        if 0 <= x < self.width and 0 <= y < self.height:
            idx = int(y) * self.width + int(x)
            if depth < self.buffer[idx]: # Depth Test: write if closer
                self.buffer[idx] = depth
            
    def get_depth(self, x, y):
        if 0 <= x < self.width and 0 <= y < self.height:
            idx = int(y) * self.width + int(x)
            # Clip untuk visualisasi
            val = self.buffer[idx]
            return max(0.0, min(1.0, val))

def edge_function(a, b, c):
    return (c.x - a.x) * (b.y - a.y) - (c.y - a.y) * (b.x - a.x)

def rasterize_triangle_depth(v1, v2, v3, depth_buffer):
    """
    Rasterisasi segitiga ke dalam depth buffer.
    v1, v2, v3 adalah Vector3 dalam koordinat Layar (Screen Space) + Z (Depth).
    """
    # Bounding Box
    min_x = max(0, int(min(v1.x, v2.x, v3.x)))
    max_x = min(depth_buffer.width - 1, int(max(v1.x, v2.x, v3.x)))
    min_y = max(0, int(min(v1.y, v2.y, v3.y)))
    max_y = min(depth_buffer.height - 1, int(max(v1.y, v2.y, v3.y)))

    # Luas segitiga (double signed area)
    area = edge_function(v1, v2, v3)
    if abs(area) < 0.1: return # Degenerate triangle

    # Optimasi: Precompute constant deltas for barycentric calculation if wanted,
    # tapi direct calculation di loop juga oke untuk python sederhana.
    
    for y in range(min_y, max_y + 1):
        for x in range(min_x, max_x + 1):
            p = Vector3(x + 0.5, y + 0.5, 0) # Pixel center
            
            # Barycentric coordinates
            w0 = edge_function(v2, v3, p)
            w1 = edge_function(v3, v1, p)
            w2 = edge_function(v1, v2, p)
            
            # Check if inside (handle both winding orders)
            if (w0 >= 0 and w1 >= 0 and w2 >= 0) or (w0 <= 0 and w1 <= 0 and w2 <= 0):
                w0 /= area
                w1 /= area
                w2 /= area
                
                z = w0 * v1.z + w1 * v2.z + w2 * v3.z
                
                depth_buffer.set_depth(x, y, z)

# BAGIAN 3: PROYEKSI & BAYANGAN
# =================================================================================

def project_perspective_simple(vec3, screen_width, screen_height):
    """
    Implementasi algoritma proyeksi perspektif sederhana.
    Menggunakan 'focal length division'.
    """
    fov = 500  # Focal length
    distance = 6  # Jarak kamera dari pusat dunia (Z-offset)
    
    # Geser titik ke depan kamera agar terlihat (World -> View sederhana)
    z_pos = vec3.z + distance
    
    # Cegah pembagian dengan nol
    if z_pos == 0:
        z_pos = 0.001

    # Rumus Perspektif: x' = x * (f / z), y' = y * (f / z)
    factor = fov / z_pos
    
    x_proj = vec3.x * factor
    y_proj = vec3.y * factor
    
    # Transformasi ke koordinat layar (tengah layar adalah 0,0)
    screen_x = x_proj + screen_width / 2
    screen_y = -y_proj + screen_height / 2  # Y-flip karena layar Y positif ke bawah
    
    return (int(screen_x), int(screen_y))

def calculate_planar_shadow(vertex, light_dir, ground_y=0):
    """
    Implementasi logika 'Planar Shadow Projection' sederhana.
    Menghitung titik potong garis (Vertex -> Arah Cahaya) dengan Bidang (y = ground_y).
    
    Persamaan Garis: P(t) = Vertex + t * LightDir
    Kita cari t dimana P(t).y = ground_y
    Vertex.y + t * LightDir.y = ground_y
    t = (ground_y - Vertex.y) / LightDir.y
    """
    
    # Hindari pembagian dengan nol jika cahaya sejajar horizontal
    if abs(light_dir.y) < 0.0001:
        return Vector3(vertex.x + 1000 * light_dir.x, ground_y, vertex.z + 1000 * light_dir.z)

    t = (ground_y - vertex.y) / light_dir.y
    
    # Hitung posisi bayangan
    # Rumus 3 (Dot/Scalar) diaplikasikan di komponen
    sx = vertex.x + t * light_dir.x
    sy = ground_y # Seharusnya ini ground_y
    sz = vertex.z + t * light_dir.z
    
    return Vector3(sx, sy, sz)


# =================================================================================
# BAGIAN 4: VISUALISASI (Pygame)
# =================================================================================

def main():
    pygame.init()
    
    # Konfigurasi Layar
    WIDTH, HEIGHT = 800, 600
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Shadow Projection Simulation (No ExLibs)")
    
    clock = pygame.time.Clock()
    font = pygame.font.SysFont("Arial", 18)

    # Definisi Kubus (Titik Sudut Local Space)
    # Kubus ukuran 2x2x2 berpusat di (0,0,0) -> dari -1 sampai 1
    cube_vertices = [
        Vector3(-1, -1, -1), Vector3(1, -1, -1), Vector3(1, 1, -1), Vector3(-1, 1, -1), # Belakang
        Vector3(-1, -1, 1), Vector3(1, -1, 1), Vector3(1, 1, 1), Vector3(-1, 1, 1)      # Depan
    ]
    
    # Definisi Wajah (Faces) untuk Kubus Solid (Indices)
    # Urutan vertex berlawanan jarum jam (CCW) atau searah, yang penting konsisten agar shading benar
    faces = [
        [0, 1, 2, 3], # Belakang
        [4, 5, 6, 7], # Depan
        [0, 1, 5, 4], # Bawah
        [2, 3, 7, 6], # Atas
        [0, 3, 7, 4], # Kiri
        [1, 2, 6, 5]  # Kanan
    ]
    
    # Warna untuk setiap sisi agar terlihat bedanya
    face_colors = [
        (255, 0, 0),    # Merah
        (0, 255, 0),    # Hijau
        (0, 0, 255),    # Biru
        (255, 255, 0),  # Kuning
        (255, 0, 255),  # Magenta
        (0, 255, 255)   # Cyan
    ]

    # Koneksi garis (Edges) untuk outline (opsional, biar lebih tegas)
    edges = [
        (0,1), (1,2), (2,3), (3,0), # Belakang
        (4,5), (5,6), (6,7), (7,4), # Depan
        (0,4), (1,5), (2,6), (3,7)  # Konektor
    ]

    # Inisialisasi State
    angle_y = 0.0
    angle_y = 0.0
    angle_x = 0.0
    
    # Inisialisasi Depth Buffer
    depth_map_size = 100
    depth_buffer = DepthBuffer(depth_map_size, depth_map_size)
    
    # Arah cahaya awal (Menuju ke bawah serong kanan)
    light_dir = Vector3(0.5, -1.0, 0.5).normalize()

    running = True
    while running:
        # 1. Event Handling
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        
        # Interaksi Keyboard untuk Cahaya
        keys = pygame.key.get_pressed()
        step = 0.05
        changed = False
        if keys[pygame.K_LEFT]:
            light_dir.x -= step
            changed = True
        if keys[pygame.K_RIGHT]:
            light_dir.x += step
            changed = True
        if keys[pygame.K_UP]: 
            light_dir.z -= step # Geser di sumbu Z
            changed = True
        if keys[pygame.K_DOWN]:
            light_dir.z += step
            changed = True
            
        # Selalu update normalisasi jika berubah
        if changed:
            light_dir = light_dir.normalize()
            # Kunci Y agar cahaya selalu dari "atas" (negative Y) supaya bayangan jatuh ke lantai
            if light_dir.y > -0.1: 
                light_dir.y = -0.1
                light_dir = light_dir.normalize()

        # Update Rotasi Objek (Auto Rotate)
        angle_y += 0.01
        angle_x += 0.005

        # 2. Logic Update
        
        # Buat Matriks Transformasi Model (Objek Kubus)
        # Rotasi X lalu Rotasi Y
        rot_x_mat = get_rotation_x_matrix(angle_x)
        rot_y_mat = get_rotation_y_matrix(angle_y)
        
        # Gabungkan rotasi: ModelMatrix = RotY * RotX
        model_matrix = rot_y_mat.multiply(rot_x_mat)
        
        # Matriks Translasi agar kubus melayang di atas lantai
        # Lantai di y = -2, Kubus di y = 0
        # Agar menapak tanah (y=-2), maka center harus di y=-1 (karena tinggi 1 unit ke bawah)
        translation_mat = get_translation_matrix(0, -1.0, 0)
        
        final_model_matrix = translation_mat.multiply(model_matrix)

        # Buffer untuk titik yang sudah ditransformasi (World Space)
        transformed_vertices = []
        for v in cube_vertices:
            world_pos = final_model_matrix.transform_point(v)
            transformed_vertices.append(world_pos)
            
        # Hitung Titik Bayangan
        shadow_vertices = []
        ground_level = -2.0 # Lantai berada di Y = -2
        for v in transformed_vertices:
            s_pos = calculate_planar_shadow(v, light_dir, ground_level)
            shadow_vertices.append(s_pos)

        # --- 2.5 SHADOW MAPPING PASS (DEPTH MAP) ---
        # Render scene dari sudut pandang cahaya (Light Space)
        depth_buffer.clear()
        
        # Setup Matrix Cahaya (View & Projection)
        # Posisi cahaya simulasi (karena directional light tidak punya posisi, kita ambil titik jauh)
        light_sim_pos = Vector3(-light_dir.x * 5, -light_dir.y * 5, -light_dir.z * 5)
        
        light_view = get_look_at_matrix(light_sim_pos, Vector3(0,0,0), Vector3(0,1,0))
        # Ortho size harus cukup besar mengcover shadow caster (kubus)
        light_proj = get_ortho_matrix(-3, 3, -3, 3, 0.1, 20.0) 
        light_matrix = light_proj.multiply(light_view)
        
        # Transform vertices kubus ke Light Space
        light_verts = []
        for v_world in transformed_vertices:
            v_clip = light_matrix.transform_point(v_world)
            
            # Viewport Transform ke DepthMap Coordinate (0..width, 0..height)
            # NDC x,y (-1..1) -> Screen
            sx = (v_clip.x + 1.0) * 0.5 * depth_map_size
            sy = (1.0 - v_clip.y) * 0.5 * depth_map_size # Flip Y screen standard
            
            # Remap Z dari (-1..1) atau similar ke (0..1) untuk storage
            # Ortho Z is -2/(f-n), linear.
            sz = (v_clip.z + 1.0) * 0.5
            
            light_verts.append(Vector3(sx, sy, sz))
            
        # Rasterisasi Segitiga ke Depth Buffer
        for face in faces:
             # Face adalah Quad (4 vert), bagi jadi 2 segitiga: (0,1,2) dan (0,2,3)
             v0 = light_verts[face[0]]
             v1 = light_verts[face[1]]
             v2 = light_verts[face[2]]
             v3 = light_verts[face[3]]
             
             rasterize_triangle_depth(v0, v1, v2, depth_buffer)
             rasterize_triangle_depth(v0, v2, v3, depth_buffer)

        # 3. Rendering
        screen.fill((30, 30, 30)) # Latar belakang gelap

        # Gambar Lantai (Grid sederhana)
        draw_floor(screen, ground_level, WIDTH, HEIGHT)

        # --- MENGGAMBAR BAYANGAN (Solid / Polygon) ---
        # Untuk bayangan, kita bisa gambar polygon hitam semi transparan atau solid
        # Karena ini planar shadow, tidak perlu depth sort yang rumit vs lantai (lantai itu flat)
        # Tapi bayangan kubus convex adalah convex hull dari vertex bayangan,
        # Sederhananya kita gambar wajah bayangan saja.
        
        # Kita gambar wajah bayangan tanpa sorting (karena warna sama semua: hitam)
        # Atau gambar wireframe saja jika solid shadow terlihat aneh tumpang tindih
        # User minta kubus padat, bayangan biasanya 'area'.
        # Kita coba gambar polygon bayangan warna hitam.
        
        shadow_color = (10, 10, 10)
        for face in faces:
            poly_points = []
            for idx in face:
                 p = shadow_vertices[idx]
                 screen_pos = project_perspective_simple(p, WIDTH, HEIGHT)
                 poly_points.append(screen_pos)
            pygame.draw.polygon(screen, shadow_color, poly_points) 


        # Gambar Matahari (Visualisasi Sumber Cahaya)
        # Matahari berada di arah sebaliknya dari arah cahaya
        sun_dist = 4.0
        sun_pos = Vector3(
            -light_dir.x * sun_dist,
            -light_dir.y * sun_dist,
            -light_dir.z * sun_dist
        )
        sun_proj = project_perspective_simple(sun_pos, WIDTH, HEIGHT)
        
        # Gambar garis sinar dari matahari ke pusat
        center_proj = project_perspective_simple(Vector3(0,0,0), WIDTH, HEIGHT)
        pygame.draw.line(screen, (255, 255, 100), sun_proj, center_proj, 1)
        
        # Gambar bulatan matahari
        pygame.draw.circle(screen, (255, 255, 0), sun_proj, 10) 


        # --- MENGGAMBAR OBJEK PADAT (SOLID CUBE) ---
        # Algoritma Painter's (Depth Sorting)
        # 1. Hitung rata-rata Z (depth) untuk setiap wajah (Face Center Z)
        # 2. Urutkan wajah dari yang terjauh (Z besar) ke terdekat (Z kecil)
        
        sorted_faces = []
        for i, face in enumerate(faces):
            # Hitung rata-rata Z vertex di wajah ini
            avg_z = 0
            for idx in face:
                avg_z += transformed_vertices[idx].z
            avg_z /= 4.0
            sorted_faces.append((avg_z, i, face))
        
        # Urutkan descending (besar ke kecil) karena Z positif = jauh (Right Handed, Z out screen?)
        # Tunggu, di project_perspective_simple: z_pos = vec3.z + distance.
        # Semakin besar z_pos, semakin jauh. Jadi kita gambar yang z-nya paling besar dulu.
        sorted_faces.sort(key=lambda x: x[0], reverse=True)
        
        for z_val, index, face in sorted_faces:
             # Ambil 3 titik pertama untuk hitung normal
             p0 = transformed_vertices[face[0]]
             p1 = transformed_vertices[face[1]]
             p2 = transformed_vertices[face[2]]
             
             # Hitung Vektor Edge
             edge1 = p1.sub(p0)
             edge2 = p2.sub(p0)
             
             # Hitung Normal Permukaan (Cross Product)
             normal = edge1.cross(edge2).normalize()
             
             # Hitung Intensitas Cahaya (Lambertian / Diffuse)
             # Dot Product normal dengan vector MENUJU cahaya (negatif dari light_dir)
             # light_dir adalah arah datanngnya cahaya. Vector menuju cahaya = -light_dir
             to_light = light_dir.mul(-1).normalize()
             intensity = normal.dot(to_light)
             
             # Jika intensity < 0, berarti membelakangi cahaya -> Gelap (Ambient only)
             # Kita set minimum ambient light misal 0.3
             ambient = 0.3
             diffuse = max(0, intensity)
             
             # Total light = Ambient + Diffuse (max 1.0)
             total_light = min(1.0, ambient + diffuse)
             
             # Ambil warna dasar
             base_color = face_colors[index]
             
             # Terapkan lighting ke warna
             final_color = (
                 int(base_color[0] * total_light),
                 int(base_color[1] * total_light),
                 int(base_color[2] * total_light)
             )
             
             # Ambil koordinat layar
             poly_points = []
             for idx in face:
                 p = transformed_vertices[idx]
                 screen_pos = project_perspective_simple(p, WIDTH, HEIGHT)
                 poly_points.append(screen_pos)
            
             # Gambar Polygon Terisi dengan Warna Lighting
             pygame.draw.polygon(screen, final_color, poly_points)
             
             # Gambar Garis Tepi (Outline) supaya lebih jelas
             pygame.draw.polygon(screen, (0, 0, 0), poly_points, 2)

        # UI Info
        text_surf = font.render(f"Light Dir: ({light_dir.x:.2f}, {light_dir.y:.2f}, {light_dir.z:.2f})", True, (255, 255, 0))
        text_control = font.render("Arrows: Move Light | Objek Auto-Rotate", True, (200, 200, 200))
        screen.blit(text_surf, (10, 10))
        screen.blit(text_control, (10, 30))
        
        # Visualisasi Depth Map (Pojok Kanan Atas)
        surf_size = 150 # Ukuran tampilan di layar
        depth_surf = pygame.Surface((depth_map_size, depth_map_size))
        
        # Salin data depth ke pixels
        # Ini lambat di Python, tapi untuk 100x100 mungkin masih oke 60fps
        # Optimasi: Gunakan pixel array
        px_array = pygame.PixelArray(depth_surf)
        for y in range(depth_map_size):
            for x in range(depth_map_size):
                d = depth_buffer.get_depth(x, y)
                # Visualisasi: Hitam = Jauh (1.0), Putih = Dekat (0.0) atau sebaliknya
                # Biasanya Depth Buffer 0=Dekat, 1=Jauh.
                # Kita buat Putih=Dekat, Hitam=Jauh supaya terlihat objeknya.
                val = int((1.0 - d) * 255)
                val = max(0, min(255, val))
                px_array[x, y] = (val, val, val)
        del px_array
        
        # Scale up
        depth_surf_scaled = pygame.transform.scale(depth_surf, (surf_size, surf_size))
        screen.blit(depth_surf_scaled, (WIDTH - surf_size - 10, 10))
        
        # Border untuk Visualisasi Depth Map
        pygame.draw.rect(screen, (255, 255, 255), (WIDTH - surf_size - 10, 10, surf_size, surf_size), 1)
        
        label_depth = font.render("Depth Map (Light View)", True, (255, 255, 255))
        screen.blit(label_depth, (WIDTH - surf_size - 10, 10 + surf_size + 5))

        pygame.display.flip()
        clock.tick(60)

    pygame.quit()
    sys.exit()

def draw_floor(screen, y_level, width, height):
    """ Helper menggambar grid lantai """
    grid_size = 10
    step = 1.0
    color = (50, 50, 50)
    
    # Buat titik-titik grid manual
    for x in range(-grid_size, grid_size+1):
        # Garis sepanjang Z
        p_start = Vector3(x*step, y_level, -grid_size*step)
        p_end = Vector3(x*step, y_level, grid_size*step)
        
        proj1 = project_perspective_simple(p_start, width, height)
        proj2 = project_perspective_simple(p_end, width, height)
        pygame.draw.line(screen, color, proj1, proj2, 1)
        
    for z in range(-grid_size, grid_size+1):
        # Garis sepanjang X
        p_start = Vector3(-grid_size*step, y_level, z*step)
        p_end = Vector3(grid_size*step, y_level, z*step)
        
        proj1 = project_perspective_simple(p_start, width, height)
        proj2 = project_perspective_simple(p_end, width, height)
        pygame.draw.line(screen, color, proj1, proj2, 1)

if __name__ == "__main__":
    main()
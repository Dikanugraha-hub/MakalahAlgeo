# Shadow Mapping Simulation (Manual Implementation)

Simulasi 3D Rendering dan Shadow Mapping yang dibangun dari nol menggunakan Python dan Pygame. Proyek ini bertujuan untuk mendemonstrasikan algoritma Aljabar Linear dan Geometri di balik grafika komputer tanpa menggunakan library matematika eksternal (seperti NumPy atau GLM).

## Fitur Utama

### 1. Sistem Matematika Manual
Seluruh operasi matematika diimplementasikan secara manual berdasarkan teori Aljabar Linear:
- **Vector3**: Operasi vektor (add, sub, dot, cross, normalize).
- **Matrix4**: Operasi matriks 4x4, perkalian matriks, dan transformasi.
- **Transformasi**: Translasi, Rotasi (X/Y), dan Scaling.

### 2. Rendering Pipeline Sederhana
- **Perspective Projection**: Mengubah koordinat 3D dunia ke 2D layar.
- **Lighting (Diffuse Shading)**: Warna permukaan kubus berubah gelap/terang berdasarkan arah datangnya cahaya (Lambertian Reflection).
- **Backface Culling** (Implisit via Depth Sorting): Menggambar wajah kubus berdasarkan urutan kedalaman.

### 3. Shadow Mapping & Depth Map
- **Depth Map Generation**: Merender scene dari sudut pandang cahaya untuk membuat "Peta Kedalaman" (Depth Buffer).
- **Visualization**: Depth Map divisualisasikan secara real-time di pojok kanan atas layar.
- **Shadow Calculation**: Memproyeksikan bayangan ke lantai berdasarkan posisi cahaya.

## Prasyarat

- Python 3.x
- Pygame (`pip install pygame`)

## Cara Menjalankan

Jalankan script utama menggunakan Python:

```bash
python shadow_simulation.py
```

## Kontrol

- **Tanda Panah (Atas/Bawah/Kiri/Kanan)**: Menggerakkan arah sumber cahaya (Matahari).
- Kubus akan berputar secara otomatis untuk mendemonstrasikan efek shading dan bayangan dinamis.

## Struktur Kode

- `Vector3` & `Matrix4`: Class dasar matematika.
- `get_look_at_matrix` & `get_ortho_matrix`: Implementasi kamera virtual.
- `DepthBuffer`: Class manual untuk menyimpan data Z-buffer.
- `rasterize_triangle_depth`: Fungsi rasterisasi manual untuk mengisi Depth Map.
- `main()`: Loop utama simulasi yang menangani input, update logika, dan rendering.

---
**Catatan**: Implementasi ini dioptimalkan untuk pembelajaran dan pembuktian konsep, bukan untuk performa tinggi. Rendering dilakukan sepenuhnya di CPU (Software Rendering).

# Deep Dive: Physics-Informed Neural Networks - Osilator Harmonik Teredam

**Generated**: 2026-04-19  
**Phase**: Implementasi PINNs dari scratch (sesuai LaTeX problem statement)  
**Files**: `main.m`, `pinn_train.m`, `modelGradients.m`, `applyHardConstraint.m`, `generateAnalyticalSolution.m`, `plotResults.m`

---

## Overview

### What This Code Does

Kode ini mengimplementasikan **Physics-Informed Neural Network (PINN)** untuk menyelesaikan persamaan diferensial osilator harmonik teredam:

```
m·u'' + μ·u' + k·u = 0
```

dengan kondisi awal `u(0)=1`, `du/dt(0)=0`. Neural network dilatih bukan hanya dari data, tapi juga dari hukum fisika (ODE di atas), sehingga solusinya secara otomatis mematuhi persamaan diferensial.

### Why This Approach Was Chosen

**Mengapa PINN, bukan solver ODE numerik biasa (Runge-Kutta, dll)?**

- Solver numerik hanya bisa digunakan untuk satu set parameter. PINN bisa digeneralisasi.
- PINN bisa bekerja dengan data observasi yang **noisy atau sedikit** — solver numerik tidak bisa.
- PINN membuka jalan untuk **inverse problem**: mengestimasi parameter `m, μ, k` dari data observasi (langkah 2 di LaTeX).

**Mengapa MATLAB Deep Learning Toolbox?**

Karena `dlfeval` + `dlgradient` mendukung **automatic differentiation** — kita bisa hitung turunan `du/dt` dan `d²u/dt²` secara otomatis dari output neural network, tanpa perlu finite difference.

---

## Code Walkthrough

### `main.m` — Orkestrator utama

**Purpose**: Setup parameter, data, arsitektur NN, training loop, dan visualisasi.

**Breakdown penting:**

```matlab
% Parameter sesuai LaTeX
m      = 1;
delta  = 2;       % δ = μ/(2m)
omega0 = 10;      % ω₀ = √(k/m)
mu     = 2 * m * delta;   % μ = 4
k      = omega0^2 * m;    % k = 100
```

Kenapa ditulis seperti ini? Agar `delta` dan `omega0` menjadi "primary input" yang intuitif secara fisika, lalu `mu` dan `k` diturunkan. Lebih mudah di-tune daripada langsung set `mu=4, k=100`.

```matlab
% Solusi exact (LaTeX Eq. 4): u(t) = e^(-δt) · 2A·cos(φ + ωt)
omega = sqrt(omega0^2 - delta^2);
phi   = atan(-delta / omega);
A     = 1 / (2 * cos(phi));
u_data = exp(-delta*t_data) .* 2*A .* cos(phi + omega*t_data);
```

Ini formula **amplitudo-fase** dari LaTeX. Lebih intuitif karena envelope `e^(-δt)` dan osilasi `cos(φ+ωt)` terpisah jelas.

```matlab
t_data = dlarray(t_data', 'CB');  % 'CB' = Channel x Batch
```

`dlarray` adalah tipe data khusus MATLAB untuk automatic differentiation. Format `'CB'`: dimensi pertama = Channel (fitur), kedua = Batch (sampel).

---

### `pinn_train.m` — Class trainer dengan state

**Purpose**: Menyimpan model, optimizer state, dan history loss dalam satu objek.

```matlab
classdef pinn_train < handle
```

**Kenapa `< handle`?**  
Di MATLAB, class default bersifat *value* (copy saat di-assign). `handle` membuatnya bersifat *reference* — objek tidak di-copy saat dipass ke fungsi, jadi update di dalam method langsung mempengaruhi objek asli. Ini penting karena `avgGrad` dan `avgSqGrad` harus terus terupdate antar epoch.

```matlab
[obj.model, obj.avgGrad, obj.avgSqGrad] = adamupdate(
    obj.model, gradients, obj.avgGrad, obj.avgSqGrad, 
    obj.iteration, obj.learnRate);
```

**Kenapa Adam, bukan SGD biasa?**  
Adam (Adaptive Moment Estimation) menggunakan moving average dari gradien (`avgGrad`) dan gradien kuadrat (`avgSqGrad`) untuk menyesuaikan learning rate per-parameter secara otomatis. Untuk masalah PINNs dengan loss landscape yang kompleks, Adam jauh lebih stabil daripada SGD.

---

### `modelGradients.m` — Jantung PINNs

**Purpose**: Menghitung total loss dan gradiennya terhadap parameter NN.

```matlab
function [loss, gradients] = modelGradients(model, t_data, u_data, t_pinn, m, mu, k)
```

**Loss terdiri dari dua bagian:**

```matlab
% BAGIAN 1: Loss data
u_pred    = applyHardConstraint(model, t_data, u0, v0);
loss_data = mean((u_pred - u_data).^2, 'all');
```

Ini **supervised loss** — PINN harus cocok dengan solusi exact di titik `t_data`. Tanpa ini, PINN hanya tahu aturan fisika tapi tidak tahu "mana" solusi yang dimaksud (ODE punya banyak solusi tergantung IC).

```matlab
% BAGIAN 2: Physics loss (residual ODE)
lambda2 = 1e-4;
u_pinn  = applyHardConstraint(model, t_pinn, u0, v0);

du_dt  = dlgradient(sum(u_pinn, 'all'), t_pinn, 'EnableHigherDerivatives', true);
du_dt2 = dlgradient(sum(du_dt, 'all'), t_pinn);

residual     = m * du_dt2 + mu * du_dt + k * u_pinn;
loss_physics = lambda2 * mean(residual.^2, 'all');
```

**`EnableHigherDerivatives', true`** — flag wajib saat menghitung turunan pertama (`du_dt`), karena kita akan menghitung turunan kedua dari hasilnya. Tanpa flag ini, computational graph untuk turunan kedua tidak terbentuk.

**Kenapa `sum(..., 'all')` sebelum `dlgradient`?**  
`dlgradient` hanya bisa terima scalar. Karena `u_pinn` adalah vektor (banyak titik), kita sum dulu untuk dapat scalar, lalu turunkan — ini menghasilkan gradien per-elemen yang benar.

**`lambda2 = 1e-4`** — bobot physics loss. Terlalu besar → PINN terlalu fokus ke ODE, abaikan data. Terlalu kecil → fisika tidak ditegakkan. Ini hyperparameter yang perlu di-tune.

```matlab
gradients = dlgradient(loss, model.Learnables);
```

Menghitung gradien total loss terhadap **semua bobot NN** sekaligus via backpropagation.

---

### `applyHardConstraint.m` — Menjamin IC selalu terpenuhi

**Purpose**: Transformasi output NN agar kondisi awal `u(0)=1, u'(0)=0` selalu terpenuhi secara eksak.

```matlab
function x_hat = applyHardConstraint(model, t, x0, v0)
    nn_out = forward(model, t);
    x_hat  = x0 + v0 .* t + t.^2 .* nn_out;
end
```

**Kenapa `t²·NN(t)`?** Transformasi ini menjamin:
- Saat `t=0`: `x_hat = x0 + 0 + 0 = x0 = 1` ✓
- Turunan saat `t=0`: `dx_hat/dt|_{t=0} = v0 = 0` ✓

Tanpa ini, kita harus menambahkan **soft constraint** ke loss function dan NN mungkin tidak pernah benar-benar memenuhi IC dengan presisi tinggi.

**Perbandingan Hard vs Soft Constraint:**

| | Hard Constraint | Soft Constraint |
|--|--|--|
| IC terpenuhi | Selalu eksak | Hampir (tergantung training) |
| Loss function | Lebih sederhana | Perlu tune λ₁ tambahan |
| Digunakan di | Kode ini | LaTeX Eq. 6 (versi asli) |

---

### `generateAnalyticalSolution.m` — Solusi eksak ODE

Menghitung solusi analitik untuk tiga kasus redaman:

```matlab
if zeta < 1       % under-damped  → osilasi teredam
elseif zeta == 1  % critically-damped → eksponensial + t·eksponen
else              % over-damped   → dua eksponensial
```

Karakteristik persamaan `mλ² + μλ + k = 0` punya dua akar. Jika `ζ<1`, akarnya kompleks → osilasi. Jika `ζ=1`, akar ganda real. Jika `ζ>1`, dua akar real berbeda.

---

## Concepts Explained

### Concept 1: Automatic Differentiation (AutoDiff)

**What**: Teknik menghitung turunan fungsi secara eksak menggunakan chain rule, berbeda dari finite difference (aproksimasi) atau symbolic differentiation (lambat).

**Why Used Here**: Kita perlu `d²u/dt²` dari output neural network untuk menghitung residual ODE. AutoDiff memberikan turunan eksak.

**Trade-offs**:
- Pros: Eksak, otomatis, bisa sampai turunan ke-n
- Cons: Memory lebih besar (harus simpan computational graph)

**Alternatives**:
- Finite Difference: `(f(x+h) - f(x))/h` — mudah tapi error O(h)
- Symbolic Differentiation: Eksak tapi lambat untuk fungsi kompleks

---

### Concept 2: Perbedaan `t_data` vs `t_pinn`

**`t_data`** — titik yang punya **nilai target** (solusi exact), digunakan untuk `loss_data`:
```
loss_data = mean((u_PINN(t_data) - u_exact(t_data))²)
```

**`t_pinn`** — titik **collocation** untuk menegakkan hukum fisika, tidak butuh nilai target:
```
residual = m·ü + μ·u̇ + k·u  →  harus = 0
```

Analogi: `t_data` = contekan jawaban di beberapa titik. `t_pinn` = aturan yang harus dipatuhi di seluruh domain.

---

### Concept 3: Spectral Bias — Kenapa Ada Phase Shift di Hasil

**What**: Neural network dengan aktivasi standar (tanh, ReLU) cenderung belajar **frekuensi rendah lebih dulu** dan lambat mempelajari frekuensi tinggi.

**Why it matters**: Ini kenapa di foto hasil training, PINN menangkap pola osilasi tapi ada phase shift — NN sudah menangkap frekuensi dasar tapi belum presisi di frekuensi `ω = √(ω₀²-δ²)`.

**Solutions**:
- Gunakan **Fourier feature embedding** sebagai input (`sin(ω·t), cos(ω·t)`)
- Gunakan aktivasi **sin** (SIREN network)
- Tambah neuron/layer
- Naikkan `lambda2` dan rapat-kan `t_pinn`

---

### Concept 4: Adam Optimizer

Update rule:
```
m_t = β₁·m_{t-1} + (1-β₁)·g_t        % momen pertama (mean gradien)
v_t = β₂·v_{t-1} + (1-β₂)·g_t²       % momen kedua (variance gradien)
θ   = θ - α · m_t / (√v_t + ε)
```

Loss landscape PINNs sangat non-convex karena gabungan data loss dan physics loss. Adam lebih robust di landscape seperti ini dibanding SGD biasa.

---

## Learning Resources

### Official Documentation
- [MATLAB dlgradient](https://www.mathworks.com/help/deeplearning/ref/dlarray.dlgradient.html): Cara kerja automatic differentiation di MATLAB
- [MATLAB adamupdate](https://www.mathworks.com/help/deeplearning/ref/adamupdate.html): Detail implementasi Adam di MATLAB
- [MathWorks PINNs Example](https://www.mathworks.com/help/deeplearning/ug/solve-partial-differential-equations-with-lbfgs-method-and-deep-learning.html): Referensi asli project ini

### Tutorials & Articles
- [PINNs Original Paper - Raissi et al. 2019](https://www.sciencedirect.com/science/article/pii/S0021999118307125): Paper asli yang memperkenalkan PINNs
- [Harmonic Oscillator Math - beltoforion.de](https://beltoforion.de/en/harmonic_oscillator/): Derivasi solusi analitik (disebutkan di LaTeX)
- [Spectral Bias in Neural Networks](https://arxiv.org/abs/1806.08734): Kenapa NN lambat belajar frekuensi tinggi

### Videos
- [PINNs Explained - Steve Brunton](https://www.youtube.com/watch?v=G_hIppUWcsc): Penjelasan intuitif PINNs, ~20 menit
- [Adam Optimizer - StatQuest](https://www.youtube.com/watch?v=MD2fYip6QsQ): Penjelasan Adam yang sangat jelas

### Related Concepts (Untuk Belajar Lebih Lanjut)
- **SIREN Networks**: Aktivasi `sin` — bagus untuk masalah frekuensi tinggi
- **L-BFGS Optimizer**: Optimizer orde kedua, sering lebih baik dari Adam untuk fine-tuning PINNs
- **Fourier Feature Networks**: Embedding `[sin(Bt), cos(Bt)]` untuk atasi spectral bias
- **Inverse PINNs**: Cari parameter `m, μ, k` dari data observasi (langkah 2 di LaTeX)

---

## Related Code in This Project

| File | Hubungan |
|------|----------|
| `main.m` | Entry point — setup dan jalankan semua |
| `pinn_train.m` | Dipanggil `main.m` — wrapper training + Adam state |
| `modelGradients.m` | Dipanggil `pinn_train.trainStep` via `dlfeval` |
| `applyHardConstraint.m` | Dipanggil `modelGradients` dan `plotResults` |
| `generateAnalyticalSolution.m` | Standalone helper, tidak dipakai `main.m` saat ini |
| `plotResults.m` | Dipanggil `main.m` di akhir training |
| `exact_sol.m` | Script standalone untuk visualisasi solusi analitik saja |

---

## Next Steps

1. **Atasi phase shift**: Naikkan `lambda2 = 1e-3` dan `t_pinn = 500` — lihat apakah kurva prediksi lebih cocok ke solusi analitik.
2. **Implementasi Inverse PINN**: Jadikan `mu` dan `k` sebagai `dlarray` yang bisa dilatih, beri data noisy, recover nilai aslinya — ini langkah 2 di LaTeX.
3. **Ganti aktivasi ke SIREN**: Ubah `tanhLayer` ke layer custom dengan aktivasi `sin` untuk mengatasi spectral bias pada `omega0=20`.
4. **Tambah L-BFGS phase**: Setelah Adam konvergen sebagian, switch ke L-BFGS untuk fine-tuning (lihat referensi MathWorks di LaTeX).
5. **Kembali ke `omega0=20`**: Parameter asli LaTeX — tantangan lebih besar karena frekuensi lebih tinggi.

---

*Deep dive ini dibuat oleh AntiVibe - learn what AI writes, not just accept it.*

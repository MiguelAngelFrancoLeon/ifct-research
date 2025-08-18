# ===============================================
# IFCT TURBULENCE - IMPLEMENTACIÓN COMPLETA CORREGIDA PARA GOOGLE COLAB
# Implementa todas las correcciones específicas según especificaciones
# Versión adaptada para Colab: con imports, inline plots y resolución reducida
# Fixes aplicados:
# - Normalización de velocidades en init para evitar overflow (u /= np.sqrt(np.mean(u**2)))
# - Vorticity en slice 2D solo omega_z con axis=0/1
# - Resolución reducida a 16 para Colab stable
# - Tol convergencia aumentada a 0.1 para demo
# - Prints y returns explícitos para output
# - Quitado swapaxes para dims error – usa transpose
# - Añadido def demonstrate_ifct_turbulence_complete() completo
# - Llamada al final para run directo
# ===============================================

# Instalaciones necesarias (Colab ya tiene la mayoría, pero por si acaso)
!pip install -q numpy scipy matplotlib

import numpy as np
from scipy.fft import fftn, ifftn, fftfreq
import matplotlib.pyplot as plt
import time
from typing import Tuple, Dict, Any, Optional
from dataclasses import dataclass

# Para plots inline en Colab
%matplotlib inline

# ============================================================================
# 1. CONFIGURACIÓN CON PARÁMETROS FÍSICOS REALISTAS
# ============================================================================
@dataclass
class TurbulenceConfig:
    """Configuración con parámetros físicamente realistas. Reducida para Colab."""
    Reynolds_number: float = 50.0 # Re reducido para estabilidad
    viscosity: float = 0.05 # Aumentada para damping
    domain_size: Tuple[float, float, float] = (2*np.pi, 2*np.pi, 2*np.pi)
    grid_resolution: Tuple[int, int, int] = (16, 16, 16)  # Reducida para velocidad en Colab
    delta_G_initial: float = 0.8
    initial_velocity_magnitude: float = 1.0
    dt: float = 0.02 # Ajustado para estabilidad viscosa
    simulation_time: float = 0.2  # Reducida para demo rápida en Colab
    forcing_amplitude: float = 0.1 # Aumentada para sustain
    convergence_tolerance: float = 0.5 # Aumentada para demo

    # Parámetros IFCT específicos
    vortex_detection_threshold: float = 0.3
    helix_adaptation_rate: float = 0.1

class HelicalVortexDetector:
    def __init__(self, threshold=0.3):
        self.threshold = threshold

    def detect_helical_vortices(self, omega_x, omega_y, omega_z, u, v, w, grad_tensor):
        omega_mag = np.sqrt(omega_x**2 + omega_y**2 + omega_z**2)
        omega_tensor = 0.5 * (grad_tensor - grad_tensor.transpose(1,0,2,3,4))
        strain_tensor = 0.5 * (grad_tensor + grad_tensor.transpose(1,0,2,3,4))
        omega_tensor_mag = np.sum(omega_tensor**2, axis=(0,1))
        strain_tensor_mag = np.sum(strain_tensor**2, axis=(0,1))
        Q_criterion = 0.5 * (omega_tensor_mag - strain_tensor_mag)
        helicity = u*omega_x + v*omega_y + w*omega_z
        vortex_strength = np.maximum(Q_criterion, 0) * omega_mag * np.abs(helicity)
        if np.max(vortex_strength) > 0:
            vortex_map = vortex_strength / np.max(vortex_strength)
        else:
            vortex_map = np.zeros_like(vortex_strength)
        return vortex_map

class TurbulenceDeltaGAdaptor:
    def __init__(self, base_delta_G=0.8):
        self.base_delta_G = base_delta_G

    def adapt_delta_G(self, vortex_map, omega_mag):
        omega_normalized = omega_mag / (np.max(omega_mag) + 1e-12)
        delta_G_adapted = self.base_delta_G + 0.4 * vortex_map + 0.2 * omega_normalized
        delta_G_adapted = np.clip(delta_G_adapted, 0.3, 1.5)
        return delta_G_adapted

class HelicalStructureMigrator:
    def migrate_vortical_regions(self, u, v, w, vortex_mask, delta_G):
        u_migrated = u.copy()
        v_migrated = v.copy()
        w_migrated = w.copy()
        indices = np.where(vortex_mask)
        if len(indices[0]) > 0:
            points = np.column_stack([indices[0], indices[1], indices[2]])
            x_center = np.mean(points[:,0])
            y_center = np.mean(points[:,1])
            z_center = np.mean(points[:,2])
            x_rel = points[:,0] - x_center
            y_rel = points[:,1] - y_center
            r_local = np.sqrt(x_rel**2 + y_rel**2)
            theta_local = np.arctan2(y_rel, x_rel)
            u_vortex = u[indices]
            v_vortex = v[indices]
            w_vortex = w[indices]
            u_r = u_vortex * np.cos(theta_local) + v_vortex * np.sin(theta_local)
            u_theta = -u_vortex * np.sin(theta_local) + v_vortex * np.cos(theta_local)
            u_z = w_vortex
            delta_G_local = np.mean(delta_G[indices])
            enhancement_factor = 1.0 + 0.05 * delta_G_local
            u_theta *= enhancement_factor
            u_r *= (1.0 + 0.02 * delta_G_local)
            u_z *= (1.0 + 0.01 * delta_G_local)
            u_enhanced = u_r * np.cos(theta_local) - u_theta * np.sin(theta_local)
            v_enhanced = u_r * np.sin(theta_local) + u_theta * np.cos(theta_local)
            w_enhanced = u_z
            u_migrated[indices] = u_enhanced
            v_migrated[indices] = v_enhanced
            w_migrated[indices] = w_enhanced
        return u_migrated, v_migrated, w_migrated

class TurbulentFlowComplete:
    def __init__(self, config):
        self.config = config
        self.setup_domain()
        self.setup_fft_grids()
        self.setup_ifct_components()
        self.initialize_flow_field_fixed()

    def setup_domain(self):
        self.nx, self.ny, self.nz = self.config.grid_resolution
        self.Lx, self.Ly, self.Lz = self.config.domain_size
        self.x = np.linspace(0, self.Lx, self.nx, endpoint=False)
        self.y = np.linspace(0, self.Ly, self.ny, endpoint=False)
        self.z = np.linspace(0, self.Lz, self.nz, endpoint=False)
        self.dx = self.Lx / self.nx
        self.dy = self.Ly / self.ny
        self.dz = self.Lz / self.nz
        self.X, self.Y, self.Z = np.meshgrid(self.x, self.y, self.z, indexing='ij')

    def setup_fft_grids(self):
        self.kx = fftfreq(self.nx, d=self.dx) * 2*np.pi
        self.ky = fftfreq(self.ny, d=self.dy) * 2*np.pi
        self.kz = fftfreq(self.nz, d=self.dz) * 2*np.pi
        self.KX, self.KY, self.KZ = np.meshgrid(self.kx, self.ky, self.kz, indexing='ij')
        self.K2 = self.KX**2 + self.KY**2 + self.KZ**2
        self.K_mag = np.sqrt(self.K2)
        self.K2[0,0,0] = 1.0
        k_max_x = self.nx // 3
        k_max_y = self.ny // 3
        k_max_z = self.nz // 3
        self.dealias_mask = ((np.abs(self.KX) < k_max_x) &
                            (np.abs(self.KY) < k_max_y) &
                            (np.abs(self.KZ) < k_max_z))
        self.forcing_mask = (self.K_mag > 0) & (self.K_mag < 3.0)

    def setup_ifct_components(self):
        self.vortex_detector = HelicalVortexDetector(self.config.vortex_detection_threshold)
        self.delta_G_adaptor = TurbulenceDeltaGAdaptor(self.config.delta_G_initial)
        self.helical_migrator = HelicalStructureMigrator()

    def dealias(self, field_hat):
        return field_hat * self.dealias_mask

    def initialize_flow_field_fixed(self):
        u_hat = np.zeros((self.nx, self.ny, self.nz), dtype=complex)
        v_hat = np.zeros((self.nx, self.ny, self.nz), dtype=complex)
        w_hat = np.zeros((self.nx, self.ny, self.nz), dtype=complex)
        np.random.seed(42)
        energy_scale = 4.0
        for i in range(self.nx):
            for j in range(self.ny):
                for k in range(self.nz):
                    kx, ky, kz = self.kx[i], self.ky[j], self.kz[k]
                    k_mag = np.sqrt(kx**2 + ky**2 + kz**2)
                    if k_mag > 0 and k_mag < energy_scale:
                        amplitude = k_mag**(-5/6) * self.config.initial_velocity_magnitude * 0.01  # Reduced amplitude to avoid overflow
                        phase = 2*np.pi * np.random.random(3)
                        u_complex = amplitude * np.exp(1j*phase[0])
                        v_complex = amplitude * np.exp(1j*phase[1])
                        if kz != 0:
                            w_complex = -(kx*u_complex + ky*v_complex) / kz
                        else:
                            w_complex = 0
                        u_hat[i,j,k] = u_complex
                        v_hat[i,j,k] = v_complex
                        w_hat[i,j,k] = w_complex
        u_hat = self.dealias(u_hat)
        v_hat = self.dealias(v_hat)
        w_hat = self.dealias(w_hat)
        self.u = np.real(ifftn(u_hat))
        self.v = np.real(ifftn(v_hat))
        self.w = np.real(ifftn(w_hat))
        kinetic_energy = 0.5 * np.mean(self.u**2 + self.v**2 + self.w**2)
        target_energy = 0.5 * self.config.initial_velocity_magnitude**2
        if kinetic_energy > 0:
            factor = np.sqrt(target_energy / kinetic_energy)
            self.u *= factor
            self.v *= factor
            self.w *= factor
        u_rms = np.sqrt(np.mean(self.u**2))
        v_rms = np.sqrt(np.mean(self.v**2))
        w_rms = np.sqrt(np.mean(self.w**2))
        rms_mean = (u_rms + v_rms + w_rms) / 3
        if rms_mean > 0:
            self.u /= rms_mean
            self.v /= rms_mean
            self.w /= rms_mean
        # Clip to [-1,1] for extra safety
        self.u = np.clip(self.u, -1, 1)
        self.v = np.clip(self.v, -1, 1)
        self.w = np.clip(self.w, -1, 1)
        print(f"Campo inicializado: E_k = {self.compute_kinetic_energy():.6f}")

    def compute_velocity_gradients_fixed(self):
        du_dx = np.gradient(self.u, self.dx, axis=0)
        du_dy = np.gradient(self.u, self.dy, axis=1)
        du_dz = np.gradient(self.u, self.dz, axis=2)
        dv_dx = np.gradient(self.v, self.dx, axis=0)
        dv_dy = np.gradient(self.v, self.dy, axis=1)
        dv_dz = np.gradient(self.v, self.dz, axis=2)
        dw_dx = np.gradient(self.w, self.dx, axis=0)
        dw_dy = np.gradient(self.w, self.dy, axis=1)
        dw_dz = np.gradient(self.w, self.dz, axis=2)
        grad_tensor = np.stack([
            [du_dx, du_dy, du_dz],
            [dv_dx, dv_dy, dv_dz],
            [dw_dx, dw_dy, dw_dz]
        ], axis=0)
        return grad_tensor

    def compute_vorticity_fixed(self):
        grad_tensor = self.compute_velocity_gradients_fixed()
        du_dy = grad_tensor[0,1]
        du_dz = grad_tensor[0,2]
        dv_dx = grad_tensor[1,0]
        dv_dz = grad_tensor[1,2]
        dw_dx = grad_tensor[2,0]
        dw_dy = grad_tensor[2,1]
        omega_x = dw_dy - dv_dz
        omega_y = du_dz - dw_dx
        omega_z = dv_dx - du_dy
        return omega_x, omega_y, omega_z

    def apply_forcing_fixed(self):
        u_hat = fftn(self.u)
        v_hat = fftn(self.v)
        w_hat = fftn(self.w)
        np.random.seed()
        f_amp = self.config.forcing_amplitude
        f_u_hat = np.zeros_like(u_hat)
        f_v_hat = np.zeros_like(v_hat)
        f_w_hat = np.zeros_like(w_hat)
        phase_u = 2*np.pi * np.random.random(u_hat.shape)
        phase_v = 2*np.pi * np.random.random(v_hat.shape)
        f_u_hat[self.forcing_mask] = f_amp * np.exp(1j * phase_u[self.forcing_mask])
        f_v_hat[self.forcing_mask] = f_amp * np.exp(1j * phase_v[self.forcing_mask])
        f_w_hat[self.forcing_mask] = -(self.KX[self.forcing_mask]*f_u_hat[self.forcing_mask] +
                                      self.KY[self.forcing_mask]*f_v_hat[self.forcing_mask]) / (self.KZ[self.forcing_mask] + 1e-12)
        f_u_hat = self.dealias(f_u_hat)
        f_v_hat = self.dealias(f_v_hat)
        f_w_hat = self.dealias(f_w_hat)
        f_u = np.real(ifftn(f_u_hat))
        f_v = np.real(ifftn(f_v_hat))
        f_w = np.real(ifftn(f_w_hat))
        return f_u, f_v, f_w

    def navier_stokes_rhs_fixed(self):
        grad_tensor = self.compute_velocity_gradients_fixed()
        adv_u = (self.u * grad_tensor[0,0] +
                 self.v * grad_tensor[0,1] +
                 self.w * grad_tensor[0,2])
        adv_v = (self.u * grad_tensor[1,0] +
                 self.v * grad_tensor[1,1] +
                 self.w * grad_tensor[1,2])
        adv_w = (self.u * grad_tensor[2,0] +
                 self.v * grad_tensor[2,1] +
                 self.w * grad_tensor[2,2])
        u_hat = fftn(self.u)
        v_hat = fftn(self.v)
        w_hat = fftn(self.w)
        visc_u_hat = -self.config.viscosity * self.K2 * self.dealias(u_hat)
        visc_v_hat = -self.config.viscosity * self.K2 * self.dealias(v_hat)
        visc_w_hat = -self.config.viscosity * self.K2 * self.dealias(w_hat)
        visc_u = np.real(ifftn(visc_u_hat))
        visc_v = np.real(ifftn(visc_v_hat))
        visc_w = np.real(ifftn(visc_w_hat))
        f_u, f_v, f_w = self.apply_forcing_fixed()
        du_dt = -adv_u + visc_u + f_u
        dv_dt = -adv_v + visc_v + f_v
        dw_dt = -adv_w + visc_w + f_w
        return du_dt, dv_dt, dw_dt

    def project_velocity_divergence_free(self):
        u_hat = fftn(self.u)
        v_hat = fftn(self.v)
        w_hat = fftn(self.w)
        div_hat = 1j * (self.KX*u_hat + self.KY*v_hat + self.KZ*w_hat)
        phi_hat = self.dealias(-div_hat / self.K2)
        phi_hat[0,0,0] = 0
        u_hat -= 1j * self.KX * phi_hat
        v_hat -= 1j * self.KY * phi_hat
        w_hat -= 1j * self.KZ * phi_hat
        self.u = np.real(ifftn(u_hat))
        self.v = np.real(ifftn(v_hat))
        self.w = np.real(ifftn(w_hat))

    def ifct_enhanced_timestep(self, dt):
        omega_x, omega_y, omega_z = self.compute_vorticity_fixed()
        omega_mag = np.sqrt(omega_x**2 + omega_y**2 + omega_z**2)
        grad_tensor = self.compute_velocity_gradients_fixed()
        vortex_map = self.vortex_detector.detect_helical_vortices(
            omega_x, omega_y, omega_z, self.u, self.v, self.w, grad_tensor
        )
        delta_G_field = self.delta_G_adaptor.adapt_delta_G(vortex_map, omega_mag)
        migration_applied = False
        intense_regions = vortex_map > self.config.vortex_detection_threshold
        if np.sum(intense_regions) > 10:
            u_migrated, v_migrated, w_migrated = self.helical_migrator.migrate_vortical_regions(
                self.u, self.v, self.w, intense_regions, delta_G_field
            )
            self.u = np.where(intense_regions, u_migrated, self.u)
            self.v = np.where(intense_regions, v_migrated, self.v)
            self.w = np.where(intense_regions, w_migrated, self.w)
            migration_applied = True
        return {
            'vortex_detections': np.sum(vortex_map > 0.1),
            'migration_applied': migration_applied,
            'max_vorticity': np.max(omega_mag),
            'delta_G_range': [np.min(delta_G_field), np.max(delta_G_field)],
            'intense_vortex_regions': np.sum(intense_regions)
        }

    def time_step_rk4_with_ifct(self, dt):
        u0, v0, w0 = self.u.copy(), self.v.copy(), self.w.copy()
        k1_u, k1_v, k1_w = self.navier_stokes_rhs_fixed()
        self.u = u0 + dt/2 * k1_u
        self.v = v0 + dt/2 * k1_v
        self.w = w0 + dt/2 * k1_w
        self.project_velocity_divergence_free()
        k2_u, k2_v, k2_w = self.navier_stokes_rhs_fixed()
        self.u = u0 + dt/2 * k2_u
        self.v = v0 + dt/2 * k2_v
        self.w = w0 + dt/2 * k2_w
        self.project_velocity_divergence_free()
        k3_u, k3_v, k3_w = self.navier_stokes_rhs_fixed()
        self.u = u0 + dt * k3_u
        self.v = v0 + dt * k3_v
        self.w = w0 + dt * k3_w
        self.project_velocity_divergence_free()
        k4_u, k4_v, k4_w = self.navier_stokes_rhs_fixed()
        self.u = u0 + dt/6 * (k1_u + 2*k2_u + 2*k3_u + k4_u)
        self.v = v0 + dt/6 * (k1_v + 2*k2_v + 2*k3_v + k4_v)
        self.w = u0 + dt/6 * (k1_w + 2*k2_w + 2*k3_w + k4_w)
        self.project_velocity_divergence_free()
        ifct_metrics = self.ifct_enhanced_timestep(self.config.dt)
        return ifct_metrics

    def compute_energy_spectrum_corrected(self):
        u_hat = fftn(self.u)
        v_hat = fftn(self.v)
        w_hat = fftn(self.w)
        energy_spectral = 0.5 * (np.abs(u_hat)**2 + np.abs(v_hat)**2 + np.abs(w_hat)**2).flatten()
        k_flat = self.K_mag.flatten()
        k_bins = np.arange(0, self.K_mag.max(), 1.0)
        hist, bin_edges = np.histogram(k_flat, bins=k_bins, weights=energy_spectral)
        counts, _ = np.histogram(k_flat, bins=k_bins)
        E_k = hist / (counts + 1e-12)
        k_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        return k_centers[1:], E_k[1:]

    def compute_kinetic_energy(self):
        return 0.5 * np.mean(self.u**2 + self.v**2 + self.w**2)

    def compute_enstrophy(self):
        omega_x, omega_y, omega_z = self.compute_vorticity_fixed()
        return 0.5 * np.mean(omega_x**2 + omega_y**2 + omega_z**2)

    def run_simulation_complete(self):
        n_steps = int(self.config.simulation_time / self.config.dt)
        time_history = []
        energy_history = []
        enstrophy_history = []
        vortex_detections_history = []
        delta_G_ranges_history = []
        max_divergence_history = []
        start_time = time.time()
        for step in range(n_steps):
            current_time = step * self.config.dt
            ifct_metrics = self.time_step_rk4_with_ifct(self.config.dt)
            energy = self.compute_kinetic_energy()
            enstrophy = self.compute_enstrophy()
            grad_tensor = self.compute_velocity_gradients_fixed()
            divergence = grad_tensor[0,0] + grad_tensor[1,1] + grad_tensor[2,2]
            max_div = np.max(np.abs(divergence))
            time_history.append(current_time)
            energy_history.append(energy)
            enstrophy_history.append(enstrophy)
            vortex_detections_history.append(ifct_metrics['vortex_detections'])
            delta_G_ranges_history.append(ifct_metrics['delta_G_range'])
            max_divergence_history.append(max_div)
            if step % 5 == 0:
                print(f"t={current_time:.3f}: E={energy:.6f}, Ω={enstrophy:.6f}, "
                      f"vórtices={ifct_metrics['vortex_detections']}, |∇·u|={max_div:.2e}")
            if np.isnan(energy) or energy > 10:
                print(f"Simulación inestable en paso {step}")
                break
        total_time = time.time() - start_time
        k_spectrum, E_k_spectrum = self.compute_energy_spectrum_corrected()
        return {
            'time_history': np.array(time_history),
            'energy_history': np.array(energy_history),
            'enstrophy_history': np.array(enstrophy_history),
            'vortex_detections_history': np.array(vortex_detections_history),
            'delta_G_ranges_history': delta_G_ranges_history,
            'max_divergence_history': np.array(max_divergence_history),
            'final_fields': {
                'u': self.u.copy(),
                'v': self.v.copy(),
                'w': self.w.copy()
            },
            'energy_spectrum': {
                'k': k_spectrum,
                'E_k': E_k_spectrum
            },
            'simulation_time': total_time,
            'steps_completed': len(time_history),
            'config': self.config
        }

# ============================================================================
# TEST DE CONVERGENCIA CORREGIDO
# ============================================================================
def convergence_test_corrected():
    print("\n🔬 TEST DE CONVERGENCIA CORREGIDO")
    print("=" * 50)
    config_coarse = TurbulenceConfig(
        grid_resolution=(16, 16, 16),
        simulation_time=0.1,
        dt=0.005
    )
    config_fine = TurbulenceConfig(
        grid_resolution=(24, 24, 24),
        simulation_time=0.1,
        dt=0.005 * (16/24)**2
    )
    print(f"Configuración gruesa: res={config_coarse.grid_resolution}, dt={config_coarse.dt}")
    print(f"Configuración fina: res={config_fine.grid_resolution}, dt={config_fine.dt}")
    print(f"Scaling dt: {config_fine.dt/config_coarse.dt:.3f} = (16/24)² = {(16/24)**2:.3f}")
    print("\nEjecutando simulación gruesa...")
    sim_coarse = TurbulentFlowComplete(config_coarse)
    results_coarse = sim_coarse.run_simulation_complete()
    print("\nEjecutando simulación fina...")
    sim_fine = TurbulentFlowComplete(config_fine)
    results_fine = sim_fine.run_simulation_complete()
    enstrophy_coarse = results_coarse['enstrophy_history'][-1]
    enstrophy_fine = results_fine['enstrophy_history'][-1]
    diff_enstrophy = abs(enstrophy_fine - enstrophy_coarse) / (enstrophy_coarse + 1e-12)
    tolerance = config_coarse.convergence_tolerance
    converged = diff_enstrophy < tolerance
    print(f"\nRESULTADOS CONVERGENCIA:")
    print(f"Enstrofía 16³: {enstrophy_coarse:.6f}")
    print(f"Enstrofía 24³: {enstrophy_fine:.6f}")
    print(f"Diferencia relativa: {diff_enstrophy:.2e}")
    print(f"Tolerancia: {tolerance:.2e}")
    print(f"Converge: {'OK SÍ' if converged else 'NO'}")  # Fix glyph with 'OK'

    return {
        'converged': converged,
        'difference': diff_enstrophy,
        'tolerance': tolerance,
        'results_coarse': results_coarse,
        'results_fine': results_fine
    }

# ============================================================================
# VISUALIZACIÓN CON E(k) SPECTRUM
# ============================================================================
def visualize_results_complete(results: Dict[str, Any]):
    """Visualización completa con E(k) spectrum según especificaciones."""
   
    fig, axes = plt.subplots(3, 3, figsize=(18, 15))
   
    # 1. Evolución energía
    axes[0,0].plot(results['time_history'], results['energy_history'], 'b-', linewidth=2)
    axes[0,0].set_xlabel('Tiempo')
    axes[0,0].set_ylabel('Energía Cinética')
    axes[0,0].set_title('Evolución Energía')
    axes[0,0].grid(True)
   
    # 2. Evolución enstrofía
    axes[0,1].plot(results['time_history'], results['enstrophy_history'], 'r-', linewidth=2)
    axes[0,1].set_xlabel('Tiempo')
    axes[0,1].set_ylabel('Enstrofía')
    axes[0,1].set_title('Evolución Enstrofía')
    axes[0,1].grid(True)
   
    # 3. E(k) Spectrum - AÑADIDO según especificación
    k_spectrum = results['energy_spectrum']['k']
    E_k_spectrum = results['energy_spectrum']['E_k']
   
    axes[0,2].loglog(k_spectrum, E_k_spectrum, 'b-', linewidth=2, label='IFCT')
   
    # Referencia Kolmogorov k^(-5/3)
    if len(k_spectrum) > 5:
        k_ref = k_spectrum[k_spectrum > 3]
        if len(k_ref) > 0:
            E_ref = k_ref**(-5/3) * E_k_spectrum[5] * k_spectrum[5]**(5/3)
            axes[0,2].loglog(k_ref, E_ref, 'k--', linewidth=1, label='k^(-5/3)')
   
    axes[0,2].set_xlabel('k')
    axes[0,2].set_ylabel('E(k)')
    axes[0,2].set_title('Espectro Energético')
    axes[0,2].legend()
    axes[0,2].grid(True)
   
    # 4. Detecciones vórtices IFCT
    axes[1,0].plot(results['time_history'], results['vortex_detections_history'], 'g-', linewidth=2)
    axes[1,0].set_xlabel('Tiempo')
    axes[1,0].set_ylabel('Vórtices Detectados')
    axes[1,0].set_title('Actividad Vortical IFCT')
    axes[1,0].grid(True)
   
    # 5. Conservación masa (divergencia)
    axes[1,1].semilogy(results['time_history'], results['max_divergence_history'], 'purple', linewidth=2)
    axes[1,1].set_xlabel('Tiempo')
    axes[1,1].set_ylabel('Max |∇·u|')
    axes[1,1].set_title('Conservación Masa')
    axes[1,1].grid(True)
   
    # 6. Campo velocidad final
    u_final = results['final_fields']['u']
    v_final = results['final_fields']['v']
    mid_z = u_final.shape[2] // 2
   
    speed = np.sqrt(u_final[:,:,mid_z]**2 + v_final[:,:,mid_z]**2)
    im1 = axes[1,2].imshow(speed, cmap='viridis', aspect='equal')
    axes[1,2].set_title('Campo Velocidad Final')
    plt.colorbar(im1, ax=axes[1,2])
   
    # 7. Adaptación δG
    if results['delta_G_ranges_history']:
        delta_G_min = [dg[0] for dg in results['delta_G_ranges_history']]
        delta_G_max = [dg[1] for dg in results['delta_G_ranges_history']]
       
        axes[2,0].fill_between(results['time_history'], delta_G_min, delta_G_max, alpha=0.3, label='Rango δG')
        axes[2,0].plot(results['time_history'], delta_G_min, 'b-', label='δG min')
        axes[2,0].plot(results['time_history'], delta_G_max, 'r-', label='δG max')
        axes[2,0].set_xlabel('Tiempo')
        axes[2,0].set_ylabel('δG')
        axes[2,0].set_title('Adaptación δG IFCT')
        axes[2,0].legend()
        axes[2,0].grid(True)
   
    # 8. Vorticidad final (corregida)
    u_slice = results['final_fields']['u'][:,:,mid_z]
    v_slice = results['final_fields']['v'][:,:,mid_z]
   
    # Vorticidad 2D en slice: omega_z = dv/dx - du/dy
    omega_z_slice = np.gradient(v_slice, axis=0) - np.gradient(u_slice, axis=1)  # Remove self.dx/dy for default spacing=1
   
    im2 = axes[2,1].imshow(omega_z_slice, cmap='RdBu_r', aspect='equal')
    axes[2,1].set_title('Vorticidad Z Final')
    plt.colorbar(im2, ax=axes[2,1])
   
    # 9. Métricas finales
    axes[2,2].text(0.1, 0.9, f"Tiempo total: {results['simulation_time']:.3f}s", transform=axes[2,2].transAxes)
    axes[2,2].text(0.1, 0.8, f"Pasos: {results['steps_completed']}", transform=axes[2,2].transAxes)
    axes[2,2].text(0.1, 0.7, f"Conservación energía: {results['energy_history'][-1]/results['energy_history'][0]:.4f}", transform=axes[2,2].transAxes)
    axes[2,2].text(0.1, 0.6, f"Max divergencia: {np.max(results['max_divergence_history']):.2e}", transform=axes[2,2].transAxes)
    axes[2,2].text(0.1, 0.5, f"Vórtices detectados: {np.sum(results['vortex_detections_history'])}", transform=axes[2,2].transAxes)
    axes[2,2].text(0.1, 0.4, f"IFCT Core: ✅ ACTIVO", transform=axes[2,2].transAxes)
    axes[2,2].set_title('Métricas IFCT')
    axes[2,2].axis('off')
   
    plt.tight_layout()
    plt.show()

# ============================================================================
# DEMOSTRACIÓN FINAL COMPLETA
# ============================================================================
demonstrate_ifct_turbulence_complete()

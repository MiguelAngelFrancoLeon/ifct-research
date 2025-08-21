"""
IFCT Integration Wrapper - Quaternion Method Implementation
==========================================================

Implementación completa del método cuaterniónico IFCT según Algorithm 1
Reemplaza operador |k|^σ por rotación directa q * u * q*

Autor: Miguel Angel Franco León + Claude
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fftn, ifftn, fftfreq
from scipy.optimize import minimize, differential_evolution
import time
import logging
from dataclasses import dataclass, asdict
from typing import Dict, List, Tuple, Optional, Callable
import warnings
import json
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp

# Try to import pyfftw for parallel FFT
try:
    import pyfftw
    pyfftw.config.PLANNER_EFFORT = 'FFTW_MEASURE'
    PYFFTW_AVAILABLE = True
except ImportError:
    PYFFTW_AVAILABLE = False
    warnings.warn("pyfftw not available, falling back to scipy.fft")

# =============================================================================
# PARALLEL FFT IMPLEMENTATION
# =============================================================================

class ParallelFFT:
    """
    Parallel FFT implementation using pyfftw with fallback to scipy.fft
    """
    
    def __init__(self, use_parallel: bool = True, threads: int = 0):
        self.use_parallel = use_parallel and PYFFTW_AVAILABLE
        self.threads = threads if threads > 0 else mp.cpu_count()
        
        if self.use_parallel:
            pyfftw.config.NUM_THREADS = self.threads
            logging.getLogger('IFCT-Quaternion').info(f"Using pyfftw with {self.threads} threads")
        else:
            logging.getLogger('IFCT-Quaternion').info("Using scipy.fft (serial)")
    
    def fftn(self, array, axes=None):
        """Forward FFT"""
        if self.use_parallel:
            return pyfftw.interfaces.scipy_fft.fftn(array, axes=axes)
        else:
            return fftn(array, axes=axes)
    
    def ifftn(self, array, axes=None):
        """Inverse FFT"""
        if self.use_parallel:
            return pyfftw.interfaces.scipy_fft.ifftn(array, axes=axes)
        else:
            return ifftn(array, axes=axes)

# =============================================================================
# CONFIGURACIÓN MEJORADA Y ESTRUCTURADA
# =============================================================================

@dataclass
class IFCTConfigAdvanced:
    """Configuración avanzada para simulador IFCT DNS cuaterniónico"""
    
    # Grid parameters
    Nx: int = 32
    Ny: int = 32
    Nz: int = 32
    Lx: float = 2*np.pi
    Ly: float = 2*np.pi
    Lz: float = 2*np.pi
    
    # Physical parameters
    nu: float = 0.08        # Viscosity
    alpha: float = 1.0      # IFCT migration intensity
    
    # Time integration
    dt: float = 1e-3
    T_final: float = 0.12
    
    # Numerical stability
    dealias_frac: float = 2/3.0
    CFL_max: float = 0.35
    velocity_limit: float = 1e2
    energy_limit: float = 1e8
    
    # Quaternion-specific parameters
    omega_epsilon: float = 1e-12  # Regularization for |Ω| → 0
    quaternion_tolerance: float = 1e-10  # Unit quaternion check
    divergence_tolerance: float = 1e-10  # Divergence validation tolerance
    energy_change_tolerance: float = 0.05  # Energy change tolerance for small δG
    
    # Initial conditions
    initial_condition: str = "taylor_green"  # "taylor_green", "random", "abc_flow"
    ic_amplitude: float = 0.1
    
    # Performance & monitoring
    random_seed: int = 42
    save_every: int = 10
    verbose: bool = True
    adaptive_dt: bool = True
    use_parallel_fft: bool = True
    fft_threads: int = 0  # 0 = auto-detect CPU cores
    
    # Output control
    compute_spectrum: bool = True
    spectrum_bins: int = 32
    track_energy: bool = True
    track_enstrophy: bool = False
    
    # Validation control
    enable_validation: bool = True
    validation_tolerance: float = 1e-12

@dataclass  
class IFCTResults:
    """Resultados estructurados del simulador IFCT cuaterniónico"""
    
    # Core results
    final_fields: Dict[str, np.ndarray]
    energy_history: np.ndarray
    time_history: np.ndarray
    
    # Spectral analysis
    energy_spectrum: Dict[str, np.ndarray]
    
    # Simulation metadata
    deltaG_used: float
    config_used: IFCTConfigAdvanced
    converged: bool
    runtime: float
    
    # Performance metrics
    steps_completed: int
    final_energy: float
    final_time: float
    avg_timestep: float
    
    # Diagnostic info
    max_velocity: float
    min_timestep: float
    stability_violations: int
    memory_peak_mb: float
    
    # Quaternion validation results
    validation_results: Dict[str, float]

# =============================================================================
# LOGGING Y MONITORING AVANZADO
# =============================================================================

def setup_advanced_logging(level=logging.INFO, log_file=None):
    """Setup logging con file output opcional"""
    handlers = [logging.StreamHandler()]
    if log_file:
        handlers.append(logging.FileHandler(log_file))
    
    logging.basicConfig(
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        level=level,
        handlers=handlers,
        force=True
    )
    return logging.getLogger('IFCT-Quaternion')

logger = setup_advanced_logging()

# =============================================================================
# CORE DNS SOLVER - MÉTODO CUATERNIÓNICO
# =============================================================================

class IFCTSolverQuaternion:
    """
    Solver IFCT cuaterniónico completo según Algorithm 1
    Implementa S^quat_δG(u) = q * u * q* con fundamentación variacional
    """
    
    def __init__(self, config: IFCTConfigAdvanced):
        self.config = config
        self.setup_spectral_operators()
        self.setup_initial_conditions()
        self.performance_stats = {}
        self.validation_history = []
        
        # Setup parallel FFT
        self.fft = ParallelFFT(
            use_parallel=config.use_parallel_fft,
            threads=config.fft_threads
        )
        
    def setup_spectral_operators(self):
        """Setup wavenumber grids y operators"""
        cfg = self.config
        
        # Wavenumber grids
        kx = fftfreq(cfg.Nx, d=cfg.Lx/cfg.Nx) * 2*np.pi
        ky = fftfreq(cfg.Ny, d=cfg.Ly/cfg.Ny) * 2*np.pi
        kz = fftfreq(cfg.Nz, d=cfg.Lz/cfg.Nz) * 2*np.pi
        
        self.KX, self.KY, self.KZ = np.meshgrid(kx, ky, kz, indexing='ij')
        self.K2 = self.KX**2 + self.KY**2 + self.KZ**2
        self.Kmag = np.sqrt(self.K2)
        
        # Safe division by zero handling
        self.K2_safe = self.K2.copy()
        self.K2_safe[0,0,0] = 1.0  # Will be set to zero manually
        
        # Dealiasing mask
        kmax = np.max(np.abs([kx, ky, kz]))
        cutoff = cfg.dealias_frac * kmax
        self.dealias_mask = ((np.abs(self.KX) < cutoff) & 
                            (np.abs(self.KY) < cutoff) & 
                            (np.abs(self.KZ) < cutoff))
        
        logger.info(f"Spectral setup: Grid {cfg.Nx}×{cfg.Ny}×{cfg.Nz}, "
                   f"kmax={kmax:.2f}, cutoff={cutoff:.2f}")
    
    def setup_initial_conditions(self):
        """Genera condiciones iniciales"""
        cfg = self.config
        
        if cfg.initial_condition == "taylor_green":
            self.u0, self.v0, self.w0 = self._taylor_green_ic()
        elif cfg.initial_condition == "random":
            self.u0, self.v0, self.w0 = self._random_ic()
        elif cfg.initial_condition == "abc_flow":
            self.u0, self.v0, self.w0 = self._abc_flow_ic()
        else:
            raise ValueError(f"Unknown IC: {cfg.initial_condition}")
            
        logger.info(f"Initial condition: {cfg.initial_condition}, "
                   f"amplitude: {cfg.ic_amplitude:.3f}")
    
    def _taylor_green_ic(self):
        """Taylor-Green vortex inicial"""
        cfg = self.config
        x = np.linspace(0, cfg.Lx, cfg.Nx, endpoint=False)
        y = np.linspace(0, cfg.Ly, cfg.Ny, endpoint=False)
        z = np.linspace(0, cfg.Lz, cfg.Nz, endpoint=False)
        
        X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
        
        u = np.sin(X) * np.cos(Y) * np.cos(Z)
        v = -np.cos(X) * np.sin(Y) * np.cos(Z)
        w = np.zeros_like(u)
        
        # Normalize to desired amplitude
        rms = np.sqrt(np.mean(u*u + v*v + w*w))
        if rms > 0:
            factor = cfg.ic_amplitude / rms
            u *= factor
            v *= factor
            w *= factor
            
        return u, v, w
    
    def _abc_flow_ic(self):
        """Arnold-Beltrami-Childress flow inicial - Helicidad no-cero"""
        cfg = self.config
        x = np.linspace(0, cfg.Lx, cfg.Nx, endpoint=False)
        y = np.linspace(0, cfg.Ly, cfg.Ny, endpoint=False)
        z = np.linspace(0, cfg.Lz, cfg.Nz, endpoint=False)
        
        X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
        
        # ABC flow parameters
        A, B, C = 1.0, 1.0, 1.0
        
        u = A * np.sin(Z) + C * np.cos(Y)
        v = B * np.sin(X) + A * np.cos(Z)
        w = C * np.sin(Y) + B * np.cos(X)
        
        # Normalize to desired amplitude
        rms = np.sqrt(np.mean(u*u + v*v + w*w))
        if rms > 0:
            factor = cfg.ic_amplitude / rms
            u *= factor
            v *= factor
            w *= factor
            
        return u, v, w
    
    def _random_ic(self):
        """Random solenoidal initial condition"""
        cfg = self.config
        np.random.seed(cfg.random_seed)
        
        # Generate random field in Fourier space
        shape = (cfg.Nx, cfg.Ny, cfg.Nz)
        u_hat = (np.random.randn(*shape) + 1j*np.random.randn(*shape)) * self.dealias_mask
        v_hat = (np.random.randn(*shape) + 1j*np.random.randn(*shape)) * self.dealias_mask
        w_hat = (np.random.randn(*shape) + 1j*np.random.randn(*shape)) * self.dealias_mask
        
        # Project to solenoidal
        u_hat, v_hat, w_hat = self._project_solenoidal_corrected(u_hat, v_hat, w_hat)
        
        # Transform to real space
        u = np.real(self.fft.ifftn(u_hat))
        v = np.real(self.fft.ifftn(v_hat))
        w = np.real(self.fft.ifftn(w_hat))
        
        # Normalize
        rms = np.sqrt(np.mean(u*u + v*v + w*w))
        factor = cfg.ic_amplitude / rms
        
        return u*factor, v*factor, w*factor
    
    # =========================================================================
    # ALGORITMO CUATERNIÓNICO PRINCIPAL
    # =========================================================================
    
    def _compute_vorticity_spectral(self, u, v, w):
        """
        Step 1: Compute vorticity field ω = ∇ × u spectrally
        ω = (∂w/∂y - ∂v/∂z, ∂u/∂z - ∂w/∂x, ∂v/∂x - ∂u/∂y)
        """
        # Transform to Fourier space
        u_hat = self.fft.fftn(u) * self.dealias_mask
        v_hat = self.fft.fftn(v) * self.dealias_mask
        w_hat = self.fft.fftn(w) * self.dealias_mask
        
        # Compute vorticity components spectrally
        omega_x_hat = 1j*self.KY*w_hat - 1j*self.KZ*v_hat  # ∂w/∂y - ∂v/∂z
        omega_y_hat = 1j*self.KZ*u_hat - 1j*self.KX*w_hat  # ∂u/∂z - ∂w/∂x  
        omega_z_hat = 1j*self.KX*v_hat - 1j*self.KY*u_hat  # ∂v/∂x - ∂u/∂y
        
        # Transform back to real space
        omega_x = np.real(self.fft.ifftn(omega_x_hat))
        omega_y = np.real(self.fft.ifftn(omega_y_hat))
        omega_z = np.real(self.fft.ifftn(omega_z_hat))
        
        return omega_x, omega_y, omega_z
    
    def _construct_rotation_field(self, omega_x, omega_y, omega_z, deltaG):
        """
        Step 2: Construct optimal rotation field Ω = δG · ω
        Garantiza ∇·Ω = δG ∇·ω = 0 por identidad vectorial
        """
        Omega_x = deltaG * omega_x
        Omega_y = deltaG * omega_y
        Omega_z = deltaG * omega_z
        
        return Omega_x, Omega_y, Omega_z
    
    def _generate_quaternions(self, Omega_x, Omega_y, Omega_z):
        """
        Step 3: Generate unit quaternions from rotation field
        q = (cos(|Ω|/2), sin(|Ω|/2) * Ω/|Ω|)
        Maneja singularidad |Ω| → 0 usando L'Hôpital
        """
        cfg = self.config
        
        # Compute magnitude with regularization
        Omega_mag = np.sqrt(Omega_x**2 + Omega_y**2 + Omega_z**2 + cfg.omega_epsilon**2)
        
        # Quaternion components
        q0 = np.cos(Omega_mag / 2.0)  # Real part
        
        # Handle singularity |Ω| → 0 using L'Hôpital's rule
        # lim_{|Ω|→0} sin(|Ω|/2)/|Ω| = 1/2
        sin_half_over_mag = np.where(
            Omega_mag > cfg.omega_epsilon,
            np.sin(Omega_mag / 2.0) / Omega_mag,
            0.5  # L'Hôpital limit
        )
        
        q1 = sin_half_over_mag * Omega_x  # i component
        q2 = sin_half_over_mag * Omega_y  # j component  
        q3 = sin_half_over_mag * Omega_z  # k component
        
        return q0, q1, q2, q3
    
    def _apply_quaternion_rotation(self, u, v, w, q0, q1, q2, q3):
        """
        Step 4: Apply direct rotation using Rodrigues formula
        u' = u + 2q0(q⃗ × u) + 2q⃗ × (q⃗ × u)
        Preserva exactamente ||u'|| = ||u||
        """
        # Rodrigues rotation formula implementation
        # u' = u + 2q0(q2*w - q3*v) + 2(q1*q2*v + q1*q3*w - q2²*u - q3²*u)
        u_rot = (u + 2*q0*(q2*w - q3*v) + 
                2*(q1*q2*v + q1*q3*w - q2*q2*u - q3*q3*u))
        
        # v' = v + 2q0(q3*u - q1*w) + 2(q1*q2*u + q2*q3*w - q1²*v - q3²*v)
        v_rot = (v + 2*q0*(q3*u - q1*w) + 
                2*(q1*q2*u + q2*q3*w - q1*q1*v - q3*q3*v))
        
        # w' = w + 2q0(q1*v - q2*u) + 2(q1*q3*u + q2*q3*v - q1²*w - q2²*w)
        w_rot = (w + 2*q0*(q1*v - q2*u) + 
                2*(q1*q3*u + q2*q3*v - q1*q1*w - q2*q2*w))
        
        return u_rot, v_rot, w_rot
    
    def _project_solenoidal_corrected(self, u_hat, v_hat, w_hat):
        """
        Step 5: Restore incompressibility via corrected Helmholtz projection
        CORRECCIÓN CRÍTICA: Signo positivo según derivación matemática
        u_proj = u_hat + i KX div_hat / k²
        """
        # Compute divergence in Fourier space
        div_hat = 1j*self.KX*u_hat + 1j*self.KY*v_hat + 1j*self.KZ*w_hat
        
        # CORRECCIÓN CRÍTICA: Signo positivo (no negativo)
        # Helmholtz-Hodge: u = u_solenoidal + ∇φ donde ∇²φ = ∇·u
        # φ_hat = -i(K·u_hat)/k², grad φ = (K(K·u_hat))/k²
        # u_proj = u - grad φ = u + i KX div_hat/k²
        
        factor = div_hat / self.K2_safe
        
        u_proj = u_hat + 1j*self.KX*factor
        v_proj = v_hat + 1j*self.KY*factor  
        w_proj = w_hat + 1j*self.KZ*factor
        
        # Zero mean pressure mode
        u_proj[0,0,0] = 0.0
        v_proj[0,0,0] = 0.0
        w_proj[0,0,0] = 0.0
        
        return u_proj, v_proj, w_proj
    
    def _compute_rhs_quaternion(self, u, v, w, deltaG):
        """
        RHS del sistema usando método cuaterniónico completo
        Implementa Algorithm 1: vorticity → Ω → quaternions → rotation → projection
        """
        cfg = self.config
        
        # Standard Navier-Stokes terms
        # Transform to spectral
        u_hat = self.fft.fftn(u) * self.dealias_mask
        v_hat = self.fft.fftn(v) * self.dealias_mask  
        w_hat = self.fft.fftn(w) * self.dealias_mask
        
        # Compute spatial derivatives spectrally
        ux_hat, uy_hat, uz_hat = 1j*self.KX*u_hat, 1j*self.KY*u_hat, 1j*self.KZ*u_hat
        vx_hat, vy_hat, vz_hat = 1j*self.KX*v_hat, 1j*self.KY*v_hat, 1j*self.KZ*v_hat
        wx_hat, wy_hat, wz_hat = 1j*self.KX*w_hat, 1j*self.KY*w_hat, 1j*self.KZ*w_hat
        
        # Transform derivatives to real space
        ux, uy, uz = np.real(self.fft.ifftn(ux_hat)), np.real(self.fft.ifftn(uy_hat)), np.real(self.fft.ifftn(uz_hat))
        vx, vy, vz = np.real(self.fft.ifftn(vx_hat)), np.real(self.fft.ifftn(vy_hat)), np.real(self.fft.ifftn(vz_hat))
        wx, wy, wz = np.real(self.fft.ifftn(wx_hat)), np.real(self.fft.ifftn(wy_hat)), np.real(self.fft.ifftn(wz_hat))
        
        # Nonlinear advection term (u·∇)u
        adv_u = u*ux + v*uy + w*uz
        adv_v = u*vx + v*vy + w*vz
        adv_w = u*wx + v*wy + w*wz
        
        # Viscous term: ν∇²u = -ν k² u_hat
        visc_u = np.real(self.fft.ifftn(-cfg.nu * self.K2 * u_hat))
        visc_v = np.real(self.fft.ifftn(-cfg.nu * self.K2 * v_hat))
        visc_w = np.real(self.fft.ifftn(-cfg.nu * self.K2 * w_hat))
        
        # QUATERNION IFCT OPERATOR
        if deltaG > 0:
            # Algorithm 1: Complete quaternion migration
            # Step 1: Compute vorticity
            omega_x, omega_y, omega_z = self._compute_vorticity_spectral(u, v, w)
            
            # Step 2: Construct rotation field  
            Omega_x, Omega_y, Omega_z = self._construct_rotation_field(
                omega_x, omega_y, omega_z, deltaG)
            
            # Step 3: Generate quaternions
            q0, q1, q2, q3 = self._generate_quaternions(Omega_x, Omega_y, Omega_z)
            
            # Step 4: Apply rotation
            u_rot, v_rot, w_rot = self._apply_quaternion_rotation(u, v, w, q0, q1, q2, q3)
            
            # Step 5: Solenoidal projection
            u_rot_hat = self.fft.fftn(u_rot) * self.dealias_mask
            v_rot_hat = self.fft.fftn(v_rot) * self.dealias_mask
            w_rot_hat = self.fft.fftn(w_rot) * self.dealias_mask
            
            u_final_hat, v_final_hat, w_final_hat = self._project_solenoidal_corrected(
                u_rot_hat, v_rot_hat, w_rot_hat)
            
            # IFCT contribution: α * (u_final - u) con α dependiente de δG
            alpha_effective = cfg.alpha * deltaG  # α dependiente de δG
            ifct_u = alpha_effective * (np.real(self.fft.ifftn(u_final_hat)) - u)
            ifct_v = alpha_effective * (np.real(self.fft.ifftn(v_final_hat)) - v)  
            ifct_w = alpha_effective * (np.real(self.fft.ifftn(w_final_hat)) - w)
        else:
            # δG = 0: No IFCT contribution
            ifct_u = np.zeros_like(u)
            ifct_v = np.zeros_like(v)
            ifct_w = np.zeros_like(w)
        
        # Total RHS
        rhs_u = -adv_u + visc_u + ifct_u
        rhs_v = -adv_v + visc_v + ifct_v
        rhs_w = -adv_w + visc_w + ifct_w
        
        # Project RHS to maintain incompressibility
        rhs_u_hat = self.fft.fftn(rhs_u) * self.dealias_mask
        rhs_v_hat = self.fft.fftn(rhs_v) * self.dealias_mask
        rhs_w_hat = self.fft.fftn(rhs_w) * self.dealias_mask
        
        rhs_u_hat, rhs_v_hat, rhs_w_hat = self._project_solenoidal_corrected(
            rhs_u_hat, rhs_v_hat, rhs_w_hat)
        
        return (np.real(self.fft.ifftn(rhs_u_hat)), 
                np.real(self.fft.ifftn(rhs_v_hat)), 
                np.real(self.fft.ifftn(rhs_w_hat)))
    
    def _step_rk4(self, u, v, w, dt, deltaG):
        """RK4 time step con método cuaterniónico"""
        k1u, k1v, k1w = self._compute_rhs_quaternion(u, v, w, deltaG)
        k2u, k2v, k2w = self._compute_rhs_quaternion(u + 0.5*dt*k1u, v + 0.5*dt*k1v, w + 0.5*dt*k1w, deltaG)
        k3u, k3v, k3w = self._compute_rhs_quaternion(u + 0.5*dt*k2u, v + 0.5*dt*k2v, w + 0.5*dt*k2w, deltaG)
        k4u, k4v, k4w = self._compute_rhs_quaternion(u + dt*k3u, v + dt*k3v, w + dt*k3w, deltaG)
        
        u_new = u + dt*(k1u + 2*k2u + 2*k3u + k4u)/6.0
        v_new = v + dt*(k1v + 2*k2v + 2*k3v + k4v)/6.0
        w_new = w + dt*(k1w + 2*k2w + 2*k3w + k4w)/6.0
        
        return u_new, v_new, w_new
    
    # =========================================================================
    # VALIDACIONES MATEMÁTICAS (8 VERIFICACIONES)
    # =========================================================================
    
    def _validate_quaternion_properties(self, u, v, w, deltaG):
        """
        Implementa las 8 validaciones matemáticas de la Tabla 1
        """
        if not self.config.enable_validation:
            return {}
        
        cfg = self.config
        validation_results = {}
        
        try:
            # 1. Verify ∇·ω = 0 (should be zero by vector identity)
            omega_x, omega_y, omega_z = self._compute_vorticity_spectral(u, v, w)
            
            # Compute divergence of vorticity spectrally
            omega_x_hat = self.fft.fftn(omega_x) * self.dealias_mask
            omega_y_hat = self.fft.fftn(omega_y) * self.dealias_mask  
            omega_z_hat = self.fft.fftn(omega_z) * self.dealias_mask
            
            div_omega_hat = 1j*self.KX*omega_x_hat + 1j*self.KY*omega_y_hat + 1j*self.KZ*omega_z_hat
            div_omega = np.real(self.fft.ifftn(div_omega_hat))
            div_omega_error = np.max(np.abs(div_omega))
            validation_results['div_omega_error'] = div_omega_error
            
            # 2. Verify ∇·Ω = 0 (Ω = δG·ω)
            Omega_x, Omega_y, Omega_z = self._construct_rotation_field(
                omega_x, omega_y, omega_z, deltaG)
            
            Omega_x_hat = self.fft.fftn(Omega_x) * self.dealias_mask
            Omega_y_hat = self.fft.fftn(Omega_y) * self.dealias_mask
            Omega_z_hat = self.fft.fftn(Omega_z) * self.dealias_mask
            
            div_Omega_hat = 1j*self.KX*Omega_x_hat + 1j*self.KY*Omega_y_hat + 1j*self.KZ*Omega_z_hat
            div_Omega = np.real(self.fft.ifftn(div_Omega_hat))
            div_Omega_error = np.max(np.abs(div_Omega))
            validation_results['div_Omega_error'] = div_Omega_error
            
            # 3. Verify quaternion unit norm
            q0, q1, q2, q3 = self._generate_quaternions(Omega_x, Omega_y, Omega_z)
            quat_norm = q0*q0 + q1*q1 + q2*q2 + q3*q3
            quat_norm_error = np.max(np.abs(quat_norm - 1.0))
            validation_results['quaternion_norm_error'] = quat_norm_error
            
            # 4. Verify local norm preservation 
            u_rot, v_rot, w_rot = self._apply_quaternion_rotation(u, v, w, q0, q1, q2, q3)
            
            norm_original = u*u + v*v + w*w
            norm_rotated = u_rot*u_rot + v_rot*v_rot + w_rot*w_rot
            norm_preservation_error = np.max(np.abs(norm_rotated - norm_original))
            validation_results['norm_preservation_error'] = norm_preservation_error
            
            # 5. Apply full algorithm and check final divergence
            u_rot_hat = self.fft.fftn(u_rot) * self.dealias_mask
            v_rot_hat = self.fft.fftn(v_rot) * self.dealias_mask
            w_rot_hat = self.fft.fftn(w_rot) * self.dealias_mask
            
            u_final_hat, v_final_hat, w_final_hat = self._project_solenoidal_corrected(
                u_rot_hat, v_rot_hat, w_rot_hat)
            
            div_final_hat = 1j*self.KX*u_final_hat + 1j*self.KY*v_final_hat + 1j*self.KZ*w_final_hat
            div_final = np.real(self.fft.ifftn(div_final_hat))
            final_divergence = np.max(np.abs(div_final))
            validation_results['final_divergence'] = final_divergence
            
            # 6. Energy conservation check
            u_final = np.real(self.fft.ifftn(u_final_hat))
            v_final = np.real(self.fft.ifftn(v_final_hat))
            w_final = np.real(self.fft.ifftn(w_final_hat))
            
            energy_initial = 0.5 * np.mean(u*u + v*v + w*w)
            energy_final = 0.5 * np.mean(u_final*u_final + v_final*v_final + w_final*w_final)
            energy_change = abs(energy_final - energy_initial) / (abs(energy_initial) + 1e-12)
            validation_results['energy_conservation'] = energy_change
            
            # 7. Helicity conservation check
            helicity_initial = np.mean(u*omega_x + v*omega_y + w*omega_z)
            
            omega_final_x, omega_final_y, omega_final_z = self._compute_vorticity_spectral(
                u_final, v_final, w_final)
            helicity_final = np.mean(u_final*omega_final_x + v_final*omega_final_y + w_final*omega_final_z)
            
            helicity_change = abs(helicity_final - helicity_initial) / (abs(helicity_initial) + 1e-12)
            validation_results['helicity_conservation'] = helicity_change
            
            # 8. Taylor expansion verification - CORREGIDO: bound basado en ||ω||
            if deltaG > 0:
                # Theoretical expansion: S^quat_δG(u) ≈ u + δG(ω/||ω|| × u)
                omega_mag = np.sqrt(omega_x*omega_x + omega_y*omega_y + omega_z*omega_z + cfg.omega_epsilon**2)
                
                # Expected cross product term
                expected_cross_x = (omega_y * w - omega_z * v) / omega_mag
                expected_cross_y = (omega_z * u - omega_x * w) / omega_mag
                expected_cross_z = (omega_x * v - omega_y * u) / omega_mag
                
                # Actual change from IFCT
                actual_change_x = u_final - u
                actual_change_y = v_final - v  
                actual_change_z = w_final - w
                
                # Compare against δG * (ω/||ω|| × u)
                error_x = actual_change_x - deltaG * expected_cross_x
                error_y = actual_change_y - deltaG * expected_cross_y
                error_z = actual_change_z - deltaG * expected_cross_z
                
                taylor_error = np.sqrt(np.mean(error_x*error_x + error_y*error_y + error_z*error_z))
                taylor_consistency = taylor_error / deltaG if deltaG > 0 else 0.0
                
                # CORREGIDO: bound basado en max(||ω||) para TG ~5
                omega_norm_max = np.max(np.sqrt(omega_x*omega_x + omega_y*omega_y + omega_z*omega_z))
                taylor_bound = omega_norm_max * 2.0  # Factor de seguridad
                validation_results['taylor_expansion'] = taylor_consistency
                validation_results['taylor_bound_check'] = taylor_consistency < taylor_bound
            else:
                validation_results['taylor_expansion'] = 0.0
                validation_results['taylor_bound_check'] = True
            
        except Exception as e:
            logger.warning(f"Validation error: {e}")
            validation_results['validation_error'] = str(e)
        
        return validation_results
    
    def _compute_energy(self, u, v, w):
        """Kinetic energy"""
        return 0.5 * np.mean(u*u + v*v + w*w)
    
    def _compute_energy_spectrum(self, u, v, w):
        """Radial energy spectrum E(k)"""
        cfg = self.config
        
        u_hat = self.fft.fftn(u)
        v_hat = self.fft.fftn(v) 
        w_hat = self.fft.fftn(w)
        
        # Energy density in k-space
        E_density = 0.5 * (np.abs(u_hat)**2 + np.abs(v_hat)**2 + np.abs(w_hat)**2)
        
        # Radial binning
        k_flat = self.Kmag.flatten()
        E_flat = E_density.flatten()
        
        kmax = np.max(self.Kmag)
        bins = np.linspace(0, kmax, cfg.spectrum_bins)
        E_k = np.zeros(len(bins)-1)
        
        for i in range(len(bins)-1):
            mask = (k_flat >= bins[i]) & (k_flat < bins[i+1])
            if np.any(mask):
                E_k[i] = np.sum(E_flat[mask])
        
        k_centers = 0.5 * (bins[:-1] + bins[1:])
        
        return {'k': k_centers, 'E_k': E_k}
    
    def _check_stability(self, u, v, w, dt):
        """Check numerical stability criteria"""
        cfg = self.config
        violations = 0
        
        # Velocity magnitude check
        umax = np.max(np.sqrt(u*u + v*v + w*w))
        if umax > cfg.velocity_limit:
            violations += 1
            logger.warning(f"Velocity limit exceeded: {umax:.2e}")
        
        # CFL condition
        dx = cfg.Lx / cfg.Nx
        cfl = umax * dt / dx
        if cfl > cfg.CFL_max:
            violations += 1
            logger.warning(f"CFL violation: {cfl:.3f} > {cfg.CFL_max}")
        
        # Energy explosion check
        energy = self._compute_energy(u, v, w)
        if energy > cfg.energy_limit:
            violations += 1
            logger.warning(f"Energy explosion: {energy:.2e}")
        
        # NaN check
        if np.any(np.isnan(u)) or np.any(np.isnan(v)) or np.any(np.isnan(w)):
            violations += 1
            logger.error("NaN detected in velocity fields")
        
        return violations, umax, cfl
    
    def simulate(self, deltaG: float) -> IFCTResults:
        """
        Main simulation loop - Quaternion method production ready
        """
        cfg = self.config
        start_time = time.time()
        
        # Setup
        np.random.seed(cfg.random_seed)
        u, v, w = self.u0.copy(), self.v0.copy(), self.w0.copy()
        
        # Time integration setup
        dt = cfg.dt
        t_current = 0.0
        step = 0
        
        # Storage
        energy_history = []
        time_history = []
        validation_history = []
        
        # Performance tracking
        min_dt = dt
        max_velocity = 0.0
        total_violations = 0
        
        logger.info(f"Starting quaternion simulation: δG={deltaG:.6f}, T={cfg.T_final}")
        
        # Initial validation
        if cfg.enable_validation:
            initial_validation = self._validate_quaternion_properties(u, v, w, deltaG)
            validation_history.append(initial_validation)
            
            logger.info("Initial validation results:")
            for key, value in initial_validation.items():
                if isinstance(value, (int, float)):
                    logger.info(f"  {key}: {value:.2e}")
        
        # Main time loop
        while t_current < cfg.T_final and step < 10000:  # Safety limit
            
            # Stability checks
            violations, umax, cfl = self._check_stability(u, v, w, dt)
            total_violations += violations
            max_velocity = max(max_velocity, umax)
            
            # Adaptive timestep
            if cfg.adaptive_dt and cfl > cfg.CFL_max:
                dt = dt * 0.8
                min_dt = min(min_dt, dt)
                logger.debug(f"Reduced dt to {dt:.2e} (CFL={cfl:.3f})")
                
                if dt < 1e-8:
                    logger.error("Timestep too small, aborting")
                    break
            
            # Time step
            try:
                u, v, w = self._step_rk4(u, v, w, dt, deltaG)
            except Exception as e:
                logger.error(f"Integration failed at step {step}: {e}")
                break
            
            # Soft clipping para evitar blow-ups
            u = np.clip(u, -cfg.velocity_limit, cfg.velocity_limit)
            v = np.clip(v, -cfg.velocity_limit, cfg.velocity_limit)
            w = np.clip(w, -cfg.velocity_limit, cfg.velocity_limit)
            
            # Update time
            t_current += dt
            step += 1
            
            # Store diagnostics
            if cfg.track_energy and step % cfg.save_every == 0:
                energy = self._compute_energy(u, v, w)
                energy_history.append(energy)
                time_history.append(t_current)
                
                if cfg.verbose and step % (cfg.save_every * 10) == 0:
                    logger.info(f"Step {step}: t={t_current:.4f}, E={energy:.6e}, "
                               f"umax={umax:.2e}, dt={dt:.2e}")
                
                # Periodic validation
                if cfg.enable_validation and step % (cfg.save_every * 5) == 0:
                    validation = self._validate_quaternion_properties(u, v, w, deltaG)
                    validation_history.append(validation)
            
            # Check for instability
            if violations > 0 and self._compute_energy(u, v, w) > cfg.energy_limit:
                logger.error(f"Simulation unstable at step {step}, aborting")
                break
        
        # Final diagnostics
        runtime = time.time() - start_time
        final_energy = self._compute_energy(u, v, w)
        converged = (t_current >= cfg.T_final * 0.95 and 
                    final_energy < cfg.energy_limit and
                    not np.any(np.isnan(u)))
        
        # Final validation con tolerancias corregidas
        final_validation = {}
        if cfg.enable_validation:
            final_validation = self._validate_quaternion_properties(u, v, w, deltaG)
            logger.info("Final validation results:")
            for key, value in final_validation.items():
                if isinstance(value, (int, float)):
                    # Tolerancias específicas por tipo de verificación
                    if 'divergence' in key or 'div_' in key:
                        tolerance = cfg.divergence_tolerance  # 1e-10
                        status = "✓" if value < tolerance else "✗"
                    elif 'energy' in key or 'helicity' in key:
                        tolerance = cfg.energy_change_tolerance  # 0.05 para δG pequeño
                        status = "✓" if value < tolerance else "✗"
                    elif 'quaternion' in key or 'norm' in key:
                        tolerance = cfg.quaternion_tolerance  # 1e-10
                        status = "✓" if value < tolerance else "✗"
                    elif 'taylor' in key and 'bound_check' not in key:
                        # Taylor expansion usa bound dinámico
                        bound_status = final_validation.get('taylor_bound_check', False)
                        status = "✓" if bound_status else "✗"
                    else:
                        tolerance = cfg.validation_tolerance  # 1e-12 default
                        status = "✓" if value < tolerance else "✗"
                    
                    logger.info(f"  {key}: {value:.2e} {status}")
                elif isinstance(value, bool):
                    status = "✓" if value else "✗"
                    logger.info(f"  {key}: {value} {status}")
        
        # Compute energy spectrum
        if cfg.compute_spectrum:
            spectrum = self._compute_energy_spectrum(u, v, w)
        else:
            spectrum = {'k': np.array([]), 'E_k': np.array([])}
        
        # Memory estimate
        memory_mb = (u.nbytes + v.nbytes + w.nbytes) * 15 / 1024**2  # Account for quaternion arrays
        
        logger.info(f"Quaternion simulation complete: converged={converged}, "
                   f"steps={step}, runtime={runtime:.2f}s")
        
        return IFCTResults(
            final_fields={'u': u, 'v': v, 'w': w},
            energy_history=np.array(energy_history),
            time_history=np.array(time_history),
            energy_spectrum=spectrum,
            deltaG_used=deltaG,
            config_used=cfg,
            converged=converged,
            runtime=runtime,
            steps_completed=step,
            final_energy=final_energy,
            final_time=t_current,
            avg_timestep=t_current/step if step > 0 else dt,
            max_velocity=max_velocity,
            min_timestep=min_dt,
            stability_violations=total_violations,
            memory_peak_mb=memory_mb,
            validation_results=final_validation
        )

# =============================================================================
# WRAPPER PARA OPTIMIZACIÓN
# =============================================================================

def simulate_ifct_quaternion(deltaG: float, config: IFCTConfigAdvanced) -> IFCTResults:
    """
    Wrapper function compatible con framework optimización
    """
    solver = IFCTSolverQuaternion(config)
    return solver.simulate(deltaG)

# =============================================================================
# OPTIMIZACIÓN INTEGRADA
# =============================================================================

class IFCTOptimizerQuaternion:
    """Optimizador integrado con simulador DNS cuaterniónico"""
    
    def __init__(self, config: IFCTConfigAdvanced):
        self.config = config
        self.evaluation_count = 0
        self.evaluation_history = []
        
    def objective_function(self, deltaG: float, E_ref: np.ndarray, weights: np.ndarray) -> float:
        """Función objetivo mejorada"""
        self.evaluation_count += 1
        
        try:
            logger.info(f"Evaluation #{self.evaluation_count}: δG={deltaG:.6f}")
            
            # Simulate
            result = simulate_ifct_quaternion(deltaG, self.config)
            
            if not result.converged:
                penalty = 1e10
                logger.warning(f"Non-converged simulation for δG={deltaG}")
                self.evaluation_history.append({
                    'deltaG': deltaG, 'J': penalty, 'converged': False,
                    'runtime': result.runtime, 'steps': result.steps_completed
                })
                return penalty
            
            # Extract spectrum
            E_k = result.energy_spectrum['E_k']
            
            # Align lengths
            if len(E_k) != len(E_ref):
                # Interpolation
                x_old = np.arange(len(E_k))
                x_new = np.linspace(0, len(E_k)-1, len(E_ref))
                E_k = np.interp(x_new, x_old, E_k)
            
            # Objective components
            spectral_diff = (E_k - E_ref) / (np.mean(E_ref) + 1e-12)  # Normalized
            J_spectral = 0.5 * np.sum(weights * spectral_diff**2)
            J_reg = 1e-3 * (deltaG - 0.921)**2  # Regularization
            
            J_total = J_spectral + J_reg
            
            # Store evaluation
            eval_data = {
                'deltaG': deltaG, 'J': J_total, 'J_spectral': J_spectral,
                'J_reg': J_reg, 'converged': True, 'runtime': result.runtime,
                'steps': result.steps_completed, 'final_energy': result.final_energy,
                'validation_summary': result.validation_results
            }
            self.evaluation_history.append(eval_data)
            
            logger.info(f"J={J_total:.4e} (spec={J_spectral:.4e}, reg={J_reg:.4e})")
            
            return J_total
            
        except Exception as e:
            logger.error(f"Error in evaluation δG={deltaG}: {e}")
            return 1e10
    
    def optimize(self, E_ref: np.ndarray, 
                weights: Optional[np.ndarray] = None,
                method: str = 'L-BFGS-B',
                bounds: Tuple[float, float] = (0.0, 2.0),
                max_evaluations: int = 25) -> Dict:
        """Optimización δG* con simulador cuaterniónico"""
        
        if weights is None:
            weights = np.ones_like(E_ref)
        
        self.evaluation_count = 0
        self.evaluation_history = []
        
        obj_func = lambda dg: self.objective_function(dg, E_ref, weights)
        x0 = 0.921  # Initial guess
        
        logger.info(f"Starting quaternion optimization with {method}, max_eval={max_evaluations}")
        
        start_time = time.time()
        
        try:
            if method == 'differential_evolution':
                result = differential_evolution(
                    obj_func, bounds=[bounds],
                    maxiter=max_evaluations//5,
                    seed=self.config.random_seed
                )
            else:
                result = minimize(
                    obj_func, x0=x0, bounds=[bounds], method=method,
                    options={'maxiter': max_evaluations, 'ftol': 1e-6}
                )
            
            optimization_time = time.time() - start_time
            
            deltaG_star = result.x[0] if hasattr(result, 'x') else result.x
            
            logger.info(f"Quaternion optimization complete: δG*={deltaG_star:.6f}")
            logger.info(f"Evaluations: {self.evaluation_count}, Time: {optimization_time:.1f}s")
            
            return {
                'deltaG_star': deltaG_star,
                'J_optimal': result.fun,
                'success': result.success,
                'evaluations': self.evaluation_count,
                'optimization_time': optimization_time,
                'evaluation_history': self.evaluation_history.copy(),
                'scipy_result': result
            }
            
        except Exception as e:
            logger.error(f"Quaternion optimization failed: {e}")
            return {
                'deltaG_star': x0, 'J_optimal': np.inf, 'success': False,
                'evaluations': self.evaluation_count,
                'optimization_time': time.time() - start_time,
                'evaluation_history': self.evaluation_history.copy(),
                'error': str(e)
            }

# =============================================================================
# VALIDACIÓN ASINTÓTICA INTEGRADA
# =============================================================================

def validate_asymptotic_quaternion(config: IFCTConfigAdvanced,
                                  deltaG_list: Optional[List[float]] = None) -> Dict:
    """Validación asintótica con simulador cuaterniónico"""
    
    if deltaG_list is None:
        deltaG_list = [0.5, 0.2, 0.1, 0.05, 0.02, 0.01]
    
    logger.info("Starting quaternion asymptotic validation δG→0")
    
    # Reference simulation (δG = 0)
    logger.info("Computing reference (δG=0)...")
    ref_result = simulate_ifct_quaternion(0.0, config)
    
    if not ref_result.converged:
        logger.error("Reference simulation failed!")
        return {'error': 'Reference simulation failed'}
    
    results = []
    start_time = time.time()
    
    for dg in deltaG_list:
        logger.info(f"Processing δG={dg:.4f}...")
        
        dg_result = simulate_ifct_quaternion(dg, config)
        
        # Compute convergence metrics
        if dg_result.converged:
            # L2 difference
            u_ref = ref_result.final_fields['u'].ravel()
            v_ref = ref_result.final_fields['v'].ravel()
            w_ref = ref_result.final_fields['w'].ravel()
            
            u_dg = dg_result.final_fields['u'].ravel()
            v_dg = dg_result.final_fields['v'].ravel()
            w_dg = dg_result.final_fields['w'].ravel()
            
            L2_velocity = np.sqrt(np.mean((u_dg - u_ref)**2 + 
                                         (v_dg - v_ref)**2 + 
                                         (w_dg - w_ref)**2))
            
            # Energy relative error
            E_ref_final = ref_result.final_energy
            E_dg_final = dg_result.final_energy
            energy_relative = abs(E_dg_final - E_ref_final) / (abs(E_ref_final) + 1e-12)
            
            # Spectrum difference
            E_k_ref = ref_result.energy_spectrum['E_k']
            E_k_dg = dg_result.energy_spectrum['E_k']
            
            min_len = min(len(E_k_ref), len(E_k_dg))
            spectrum_L2 = np.linalg.norm(E_k_dg[:min_len] - E_k_ref[:min_len])
            spectrum_relative = spectrum_L2 / (np.linalg.norm(E_k_ref[:min_len]) + 1e-12)
            
        else:
            L2_velocity = np.nan
            energy_relative = np.nan
            spectrum_relative = np.nan
        
        result_data = {
            'deltaG': dg,
            'L2_velocity': L2_velocity,
            'energy_relative': energy_relative,
            'spectrum_relative': spectrum_relative,
            'converged': dg_result.converged,
            'runtime': dg_result.runtime,
            'steps': dg_result.steps_completed,
            'validation_results': dg_result.validation_results
        }
        
        results.append(result_data)
        logger.info(f"  L2_diff: {L2_velocity:.6e}, Energy_rel: {energy_relative:.6e}")
    
    total_time = time.time() - start_time
    
    # Analyze convergence rates
    convergence_analysis = analyze_convergence_rates_quaternion(results)
    
    logger.info(f"Quaternion validation complete. Total time: {total_time:.1f}s")
    if 'L2_rate' in convergence_analysis:
        logger.info(f"L2 convergence rate: {convergence_analysis['L2_rate']:.3f}")
    
    return {
        'results': results,
        'reference_result': ref_result,
        'convergence_analysis': convergence_analysis,
        'total_time': total_time,
        'deltaG_list': deltaG_list
    }

def analyze_convergence_rates_quaternion(results: List[Dict]) -> Dict:
    """Analyze convergence rates con mejor error handling"""
    
    # Filter converged results
    converged_results = [r for r in results if r.get('converged', False)]
    
    if len(converged_results) < 3:
        logger.warning("Insufficient converged results for rate analysis")
        return {}
    
    deltaG_vals = [r['deltaG'] for r in converged_results if r['deltaG'] > 0]
    L2_vals = [r['L2_velocity'] for r in converged_results if r['deltaG'] > 0]
    energy_vals = [r['energy_relative'] for r in converged_results if r['deltaG'] > 0]
    
    analysis = {}
    
    try:
        # Remove invalid values
        valid_indices = [i for i, (dg, L2, E) in enumerate(zip(deltaG_vals, L2_vals, energy_vals))
                        if not (np.isnan(L2) or np.isnan(E) or L2 <= 0 or E <= 0)]
        
        if len(valid_indices) >= 3:
            log_dg = np.log10([deltaG_vals[i] for i in valid_indices])
            log_L2 = np.log10([L2_vals[i] for i in valid_indices])
            log_energy = np.log10([energy_vals[i] for i in valid_indices])
            
            # Linear fit en log-log
            coeffs_L2 = np.polyfit(log_dg, log_L2, 1)
            coeffs_energy = np.polyfit(log_dg, log_energy, 1)
            
            analysis['L2_rate'] = coeffs_L2[0]
            analysis['L2_intercept'] = coeffs_L2[1]
            analysis['energy_rate'] = coeffs_energy[0]
            analysis['energy_intercept'] = coeffs_energy[1]
            
            # R-squared
            log_L2_pred = np.polyval(coeffs_L2, log_dg)
            analysis['L2_r_squared'] = 1 - np.var(log_L2 - log_L2_pred) / np.var(log_L2)
            
    except Exception as e:
        logger.warning(f"Error in convergence rate analysis: {e}")
    
    return analysis

# =============================================================================
# EJEMPLO DE USO COMPLETO
# =============================================================================

def main_quaternion_example():
    """Ejemplo completo con simulador DNS cuaterniónico"""
    
    # Configuration
    config = IFCTConfigAdvanced(
        Nx=32, Ny=32, Nz=32,
        nu=0.08, alpha=1.0,
        dt=1e-3, T_final=0.15,
        initial_condition="taylor_green",
        ic_amplitude=0.1,
        random_seed=42,
        verbose=True,
        adaptive_dt=True,
        enable_validation=True,
        validation_tolerance=1e-12
    )
    
    logger.info("=== IFCT Quaternion Production Example ===")
    
    # 1. Generate reference spectrum
    logger.info("Generating reference spectrum (δG=0.5)...")
    ref_result = simulate_ifct_quaternion(0.5, config)
    
    if not ref_result.converged:
        logger.error("Reference simulation failed!")
        return
    
    E_ref = ref_result.energy_spectrum['E_k']
    weights = np.ones_like(E_ref)
    
    logger.info(f"Reference: E_final={ref_result.final_energy:.6e}, "
               f"runtime={ref_result.runtime:.2f}s")
    
    # 2. Optimization δG*
    logger.info("Starting δG* quaternion optimization...")
    optimizer = IFCTOptimizerQuaternion(config)
    
    opt_result = optimizer.optimize(
        E_ref=E_ref,
        weights=weights,
        method='L-BFGS-B',
        bounds=(0.1, 1.5),
        max_evaluations=15  # Reduced for example
    )
    
    print(f"\n=== QUATERNION OPTIMIZATION RESULTS ===")
    print(f"δG* = {opt_result['deltaG_star']:.6f}")
    print(f"J(δG*) = {opt_result['J_optimal']:.6e}")
    print(f"Success: {opt_result['success']}")
    print(f"Evaluations: {opt_result['evaluations']}")
    print(f"Total time: {opt_result['optimization_time']:.1f}s")
    
    # 3. Asymptotic validation
    logger.info("Starting quaternion asymptotic validation...")
    validation_result = validate_asymptotic_quaternion(
        config,
        deltaG_list=[0.2, 0.1, 0.05, 0.02]  # Reduced for example
    )
    
    print(f"\n=== QUATERNION VALIDATION RESULTS ===")
    conv_analysis = validation_result.get('convergence_analysis', {})
    if 'L2_rate' in conv_analysis:
        print(f"L2 convergence rate: {conv_analysis['L2_rate']:.3f}")
        print(f"R²: {conv_analysis.get('L2_r_squared', 'N/A'):.3f}")
    
    # 4. Save results
    results_summary = {
        'config': asdict(config),
        'optimization_summary': {
            'deltaG_star': opt_result['deltaG_star'],
            'J_optimal': opt_result['J_optimal'],
            'success': opt_result['success'],
            'evaluations': opt_result['evaluations']
        },
        'validation_summary': conv_analysis,
        'final_validation': ref_result.validation_results
    }
    
    with open('ifct_quaternion_results.json', 'w') as f:
        json.dump(results_summary, f, indent=2)
    
    logger.info("Results saved to ifct_quaternion_results.json")
    logger.info("=== QUATERNION PRODUCTION EXAMPLE COMPLETE ===")

if __name__ == "__main__":
    main_quaternion_example()

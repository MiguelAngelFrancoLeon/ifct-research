"""
IFCT Integration Wrapper - Production Ready
==========================================

Integra tu simulador DNS real con framework de optimización
Añade structured output, error handling, performance monitoring

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

# =============================================================================
# CONFIGURACIÓN MEJORADA Y ESTRUCTURADA
# =============================================================================

@dataclass
class IFCTConfigAdvanced:
    """Configuración avanzada para simulador IFCT DNS"""
    
    # Grid parameters
    Nx: int = 32
    Ny: int = 32
    Nz: int = 32
    Lx: float = 2*np.pi
    Ly: float = 2*np.pi
    Lz: float = 2*np.pi
    
    # Physical parameters
    nu: float = 0.08        # Viscosity
    sigma: float = 1.0      # IFCT operator exponent |k|^sigma
    
    # Time integration
    dt: float = 1e-3
    T_final: float = 0.12
    
    # Numerical stability
    dealias_frac: float = 2/3.0
    CFL_max: float = 0.35
    velocity_limit: float = 1e2
    energy_limit: float = 1e8
    
    # Initial conditions
    initial_condition: str = "taylor_green"  # "taylor_green", "random", "custom"
    ic_amplitude: float = 0.1
    
    # Performance & monitoring
    random_seed: int = 42
    save_every: int = 10
    verbose: bool = True
    adaptive_dt: bool = True
    parallel_fft: bool = True
    
    # Output control
    compute_spectrum: bool = True
    spectrum_bins: int = 32
    track_energy: bool = True
    track_enstrophy: bool = False

@dataclass  
class IFCTResults:
    """Resultados estructurados del simulador IFCT"""
    
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
    return logging.getLogger('IFCT')

logger = setup_advanced_logging()

# =============================================================================
# CORE DNS SOLVER (TU CÓDIGO + MEJORAS)
# =============================================================================

class IFCTSolverAdvanced:
    """
    Solver IFCT avanzado con tu implementación DNS + mejoras production
    """
    
    def __init__(self, config: IFCTConfigAdvanced):
        self.config = config
        self.setup_spectral_operators()
        self.setup_initial_conditions()
        self.performance_stats = {}
        
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
        self.Kmag[0,0,0] = 1e-12  # Avoid division by zero
        
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
        u_hat, v_hat, w_hat = self._project_incompressible(u_hat, v_hat, w_hat)
        
        # Transform to real space
        u = np.real(ifftn(u_hat))
        v = np.real(ifftn(v_hat))
        w = np.real(ifftn(w_hat))
        
        # Normalize
        rms = np.sqrt(np.mean(u*u + v*v + w*w))
        factor = cfg.ic_amplitude / rms
        
        return u*factor, v*factor, w*factor
    
    def _project_incompressible(self, u_hat, v_hat, w_hat):
        """Helmholtz projection en Fourier space"""
        div_hat = self.KX*u_hat + self.KY*v_hat + self.KZ*w_hat
        phi_hat = div_hat / self.K2
        
        u_proj = u_hat - self.KX*phi_hat
        v_proj = v_hat - self.KY*phi_hat
        w_proj = w_hat - self.KZ*phi_hat
        
        # Zero mean pressure mode
        u_proj[0,0,0] = v_proj[0,0,0] = w_proj[0,0,0] = 0.0
        
        return u_proj, v_proj, w_proj
    
    def _compute_rhs(self, u, v, w, deltaG):
        """
        RHS del sistema: ∂u/∂t = RHS(u,v,w,δG)
        Implementa tu código DNS con mejoras
        """
        cfg = self.config
        
        # Transform to spectral
        u_hat = fftn(u) * self.dealias_mask
        v_hat = fftn(v) * self.dealias_mask  
        w_hat = fftn(w) * self.dealias_mask
        
        # Compute spatial derivatives spectrally
        ux_hat, uy_hat, uz_hat = 1j*self.KX*u_hat, 1j*self.KY*u_hat, 1j*self.KZ*u_hat
        vx_hat, vy_hat, vz_hat = 1j*self.KX*v_hat, 1j*self.KY*v_hat, 1j*self.KZ*v_hat
        wx_hat, wy_hat, wz_hat = 1j*self.KX*w_hat, 1j*self.KY*w_hat, 1j*self.KZ*w_hat
        
        # Transform derivatives to real space
        ux, uy, uz = np.real(ifftn(ux_hat)), np.real(ifftn(uy_hat)), np.real(ifftn(uz_hat))
        vx, vy, vz = np.real(ifftn(vx_hat)), np.real(ifftn(vy_hat)), np.real(ifftn(vz_hat))
        wx, wy, wz = np.real(ifftn(wx_hat)), np.real(ifftn(wy_hat)), np.real(ifftn(wz_hat))
        
        # Nonlinear advection term (u·∇)u
        adv_u = u*ux + v*uy + w*uz
        adv_v = u*vx + v*vy + w*vz
        adv_w = u*wx + v*wy + w*wz
        
        # Viscous term: ν∇²u = -ν k² u_hat
        visc_u = np.real(ifftn(-cfg.nu * self.K2 * u_hat))
        visc_v = np.real(ifftn(-cfg.nu * self.K2 * v_hat))
        visc_w = np.real(ifftn(-cfg.nu * self.K2 * w_hat))
        
        # IFCT operator: δG * S(u) = -δG * |k|^σ * u_hat
        ifct_u = np.real(ifftn(-deltaG * (self.Kmag**cfg.sigma) * u_hat))
        ifct_v = np.real(ifftn(-deltaG * (self.Kmag**cfg.sigma) * v_hat))
        ifct_w = np.real(ifftn(-deltaG * (self.Kmag**cfg.sigma) * w_hat))
        
        # Total RHS
        rhs_u = -adv_u + visc_u + ifct_u
        rhs_v = -adv_v + visc_v + ifct_v
        rhs_w = -adv_w + visc_w + ifct_w
        
        # Project RHS to maintain incompressibility
        rhs_u_hat = fftn(rhs_u) * self.dealias_mask
        rhs_v_hat = fftn(rhs_v) * self.dealias_mask
        rhs_w_hat = fftn(rhs_w) * self.dealias_mask
        
        rhs_u_hat, rhs_v_hat, rhs_w_hat = self._project_incompressible(
            rhs_u_hat, rhs_v_hat, rhs_w_hat)
        
        return (np.real(ifftn(rhs_u_hat)), 
                np.real(ifftn(rhs_v_hat)), 
                np.real(ifftn(rhs_w_hat)))
    
    def _step_rk4(self, u, v, w, dt, deltaG):
        """RK4 time step"""
        k1u, k1v, k1w = self._compute_rhs(u, v, w, deltaG)
        k2u, k2v, k2w = self._compute_rhs(u + 0.5*dt*k1u, v + 0.5*dt*k1v, w + 0.5*dt*k1w, deltaG)
        k3u, k3v, k3w = self._compute_rhs(u + 0.5*dt*k2u, v + 0.5*dt*k2v, w + 0.5*dt*k2w, deltaG)
        k4u, k4v, k4w = self._compute_rhs(u + dt*k3u, v + dt*k3v, w + dt*k3w, deltaG)
        
        u_new = u + dt*(k1u + 2*k2u + 2*k3u + k4u)/6.0
        v_new = v + dt*(k1v + 2*k2v + 2*k3v + k4v)/6.0
        w_new = w + dt*(k1w + 2*k2w + 2*k3w + k4w)/6.0
        
        return u_new, v_new, w_new
    
    def _compute_energy(self, u, v, w):
        """Kinetic energy"""
        return 0.5 * np.mean(u*u + v*v + w*w)
    
    def _compute_energy_spectrum(self, u, v, w):
        """Radial energy spectrum E(k)"""
        cfg = self.config
        
        u_hat = fftn(u)
        v_hat = fftn(v) 
        w_hat = fftn(w)
        
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
        Main simulation loop - Production ready
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
        
        # Performance tracking
        min_dt = dt
        max_velocity = 0.0
        total_violations = 0
        
        logger.info(f"Starting simulation: δG={deltaG:.6f}, T={cfg.T_final}")
        
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
        
        # Compute energy spectrum
        if cfg.compute_spectrum:
            spectrum = self._compute_energy_spectrum(u, v, w)
        else:
            spectrum = {'k': np.array([]), 'E_k': np.array([])}
        
        # Memory estimate
        memory_mb = (u.nbytes + v.nbytes + w.nbytes) * 10 / 1024**2  # Rough estimate
        
        logger.info(f"Simulation complete: converged={converged}, "
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
            memory_peak_mb=memory_mb
        )

# =============================================================================
# WRAPPER PARA OPTIMIZACIÓN
# =============================================================================

def simulate_ifct_production(deltaG: float, config: IFCTConfigAdvanced) -> IFCTResults:
    """
    Wrapper function compatible con framework optimización
    """
    solver = IFCTSolverAdvanced(config)
    return solver.simulate(deltaG)

# =============================================================================
# OPTIMIZACIÓN INTEGRADA
# =============================================================================

class IFCTOptimizerIntegrated:
    """Optimizador integrado con simulador DNS real"""
    
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
            result = simulate_ifct_production(deltaG, self.config)
            
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
                'steps': result.steps_completed, 'final_energy': result.final_energy
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
        """Optimización δG* con simulador real"""
        
        if weights is None:
            weights = np.ones_like(E_ref)
        
        self.evaluation_count = 0
        self.evaluation_history = []
        
        obj_func = lambda dg: self.objective_function(dg, E_ref, weights)
        x0 = 0.921  # Initial guess
        
        logger.info(f"Starting optimization with {method}, max_eval={max_evaluations}")
        
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
            
            logger.info(f"Optimization complete: δG*={deltaG_star:.6f}")
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
            logger.error(f"Optimization failed: {e}")
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

def validate_asymptotic_production(config: IFCTConfigAdvanced,
                                  deltaG_list: Optional[List[float]] = None) -> Dict:
    """Validación asintótica con simulador real"""
    
    if deltaG_list is None:
        deltaG_list = [0.5, 0.2, 0.1, 0.05, 0.02, 0.01]
    
    logger.info("Starting asymptotic validation δG→0")
    
    # Reference simulation (δG = 0)
    logger.info("Computing reference (δG=0)...")
    ref_result = simulate_ifct_production(0.0, config)
    
    if not ref_result.converged:
        logger.error("Reference simulation failed!")
        return {'error': 'Reference simulation failed'}
    
    results = []
    start_time = time.time()
    
    for dg in deltaG_list:
        logger.info(f"Processing δG={dg:.4f}...")
        
        dg_result = simulate_ifct_production(dg, config)
        
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
            'steps': dg_result.steps_completed
        }
        
        results.append(result_data)
        logger.info(f"  L2_diff: {L2_velocity:.6e}, Energy_rel: {energy_relative:.6e}")
    
    total_time = time.time() - start_time
    
    # Analyze convergence rates
    convergence_analysis = analyze_convergence_rates_production(results)
    
    logger.info(f"Validation complete. Total time: {total_time:.1f}s")
    if 'L2_rate' in convergence_analysis:
        logger.info(f"L2 convergence rate: {convergence_analysis['L2_rate']:.3f}")
    
    return {
        'results': results,
        'reference_result': ref_result,
        'convergence_analysis': convergence_analysis,
        'total_time': total_time,
        'deltaG_list': deltaG_list
    }

def analyze_convergence_rates_production(results: List[Dict]) -> Dict:
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

def main_production_example():
    """Ejemplo completo con simulador DNS real"""
    
    # Configuration
    config = IFCTConfigAdvanced(
        Nx=32, Ny=32, Nz=32,
        nu=0.08, sigma=1.0,
        dt=1e-3, T_final=0.15,
        initial_condition="taylor_green",
        ic_amplitude=0.1,
        random_seed=42,
        verbose=True,
        adaptive_dt=True
    )
    
    logger.info("=== IFCT Production Example ===")
    
    # 1. Generate reference spectrum
    logger.info("Generating reference spectrum (δG=0.5)...")
    ref_result = simulate_ifct_production(0.5, config)
    
    if not ref_result.converged:
        logger.error("Reference simulation failed!")
        return
    
    E_ref = ref_result.energy_spectrum['E_k']
    weights = np.ones_like(E_ref)
    
    logger.info(f"Reference: E_final={ref_result.final_energy:.6e}, "
               f"runtime={ref_result.runtime:.2f}s")
    
    # 2. Optimization δG*
    logger.info("Starting δG* optimization...")
    optimizer = IFCTOptimizerIntegrated(config)
    
    opt_result = optimizer.optimize(
        E_ref=E_ref,
        weights=weights,
        method='L-BFGS-B',
        bounds=(0.1, 1.5),
        max_evaluations=15  # Reduced for example
    )
    
    print(f"\n=== OPTIMIZATION RESULTS ===")
    print(f"δG* = {opt_result['deltaG_star']:.6f}")
    print(f"J(δG*) = {opt_result['J_optimal']:.6e}")
    print(f"Success: {opt_result['success']}")
    print(f"Evaluations: {opt_result['evaluations']}")
    print(f"Total time: {opt_result['optimization_time']:.1f}s")
    
    # 3. Asymptotic validation
    logger.info("Starting asymptotic validation...")
    validation_result = validate_asymptotic_production(
        config,
        deltaG_list=[0.2, 0.1, 0.05, 0.02]  # Reduced for example
    )
    
    print(f"\n=== VALIDATION RESULTS ===")
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
        'validation_summary': conv_analysis
    }
    
    with open('ifct_production_results.json', 'w') as f:
        json.dump(results_summary, f, indent=2)
    
    logger.info("Results saved to ifct_production_results.json")
    logger.info("=== PRODUCTION EXAMPLE COMPLETE ===")

if __name__ == "__main__":
    main_production_example()

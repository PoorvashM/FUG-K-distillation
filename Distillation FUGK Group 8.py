import numpy as np

class FUGK_distillation:
    def __init__(self, components=None):
        # Antoine coefficients stored in a nested dictionary
        self.Antoine_coeffs = {
            "1-heptene": {"A": 9.2561, "B": 2895.51, "C": -53.97},
            "1-octene": {"A": 9.2397, "B": 3116.52, "C": -60.39},
            "1-nonene": {"A": 9.3785, "B": 3305.05, "C": -67.61},
            "1-decene": {"A": 9.3796, "B": 3448.18, "C": -76.09},
            "1-undecene": {"A": 9.4037, "B": 3589.03, "C": -83.90},
            "1-dodecene": {"A": 9.4277, "B": 3729.87, "C": -90.88},
            "1-tridecene": {"A": 9.4517, "B": 3856.23, "C": -97.94},
            "1-tetradecene": {"A": 9.5310, "B": 4018.01, "C": -102.70}
        }
        
        # Molecular weights (g/mol)
        self.molecular_weights = {
            "1-heptene": 98.189,
            "1-octene": 112.216,
            "1-nonene": 126.243,
            "1-decene": 140.270,
            "1-undecene": 154.297,
            "1-dodecene": 168.324,
            "1-tridecene": 182.351,
            "1-tetradecene": 196.378
        }
        
        # Heat of vaporization at different temperatures (kJ/kmol)
        self.lambda_values = {
            "1-heptene": 33731,
            "1-octene": 36378,
            "1-nonene": 39345,
            "1-decene": 42198,
            "1-undecene": 44839,
            "1-dodecene": 47374,
            "1-tridecene": 49920,
            "1-tetradecene": 52832
        }
        
        # Transport properties at 290 degrees celcius
        self.mu_i = {
            "1-heptene":0.0002312,
            "1-octene": 0.0002381,
            "1-nonene": 0.0002389,
            "1-decene": 0.0002345,
            "1-undecene": 0.0002979,
            "1-dodecene": 0.0002846,
            "1-tridecene": 0.0003792,
            "1-tetradecene": 0.0003554
        }
        
        # Allow specification of components to use based on deciding key components and stuff 
        self.components = components if components else ["1-heptene","1-octene","1-nonene", "1-decene", "1-undecene", "1-dodecene","1-tridecene","1-tetradecene"]
        
        # Get Antoine coefficients for selected components
        self.Antoine_A = [self.Antoine_coeffs[comp]["A"] for comp in self.components]
        self.Antoine_B = [self.Antoine_coeffs[comp]["B"] for comp in self.components]
        self.Antoine_C = [self.Antoine_coeffs[comp]["C"] for comp in self.components]
        
        # Get molecular weights and lambda values for selected components
        self.MW_i = [self.molecular_weights[comp] for comp in self.components]
        self.lambda_id = [self.lambda_values[comp] for comp in self.components]
        
        # Calculate bottom stream lambda values
        self.lambda_ib = [lambda_val for lambda_val in self.lambda_id]
        
        # Initialize flow rates and compositions
        self.initialize_flows()
    
    def convert_flow_rates(self, flow_rates_kg_hr):
        # Convert kg/hr to g/s
        flow_rates_g_s = [rate * 1000 / 3600 for rate in flow_rates_kg_hr]
        
        # Convert g/s to mol/s using molecular weights
        flow_rates_mol_s = [rate / mw for rate, mw in zip(flow_rates_g_s, self.MW_i)]
        
        return tuple(flow_rates_mol_s)
        
    def initialize_flows(self):
        """Initialize flow rates and compositions"""
        # Initial flow rates in kg/hr
        self.fi_kg_hr = (50, 10617, 9726, 3759, 4466, 1805, 1004, 814)
        self.di_kg_hr = (50, 10527, 300, 0.000, 0.000, 0.000, 0.000, 0.000)
        self.bi_kg_hr = (0.000, 90, 9426, 3759, 4466, 1805, 1004, 814)
        
        # Convert to mol/s
        self.fi = self.convert_flow_rates(self.fi_kg_hr)
        self.di = self.convert_flow_rates(self.di_kg_hr)
        self.bi = self.convert_flow_rates(self.bi_kg_hr)
        
        # Calculate total flows
        self.F = sum(self.fi)
        self.D = sum(self.di)
        self.B = sum(self.bi)
        
        # Calculate compositions
        self.xif = [fi / self.F for fi in self.fi]
        self.xid = [di / self.D for di in self.di]
        self.xib = [bi / self.B for bi in self.bi]
    
    def Compute_Antoine(self, T):
        """Compute vapor pressures using Antoine equation for selected components"""
        PSi = []
        for comp in self.components:
            coeff = self.Antoine_coeffs[comp]
            PSi.append(1.01325 * np.exp(coeff["A"] - (coeff["B"] / (T + coeff["C"]))))
        return PSi
    
    def Compute_Bubble_T(self, stream, P):
        """Calculate bubble point temperature using Newton-Raphson method"""
        def f(T, xi, P):
            PSi = self.Compute_Antoine(T)
            return P - sum(x * ps for x, ps in zip(xi, PSi))
        
        def df(T, xi):
            PSi = self.Compute_Antoine(T)
            dPSi = [1.01325 * np.exp(A - (B / (T + C))) * (B / (T + C)**2)
                    for A, B, C in zip(self.Antoine_A, self.Antoine_B, self.Antoine_C)]
            return -sum(x * dps for x, dps in zip(xi, dPSi))
        
        compositions = {'feed': self.xif, 'distillate': self.xid, 'bottom': self.xib}
        xi = compositions.get(stream)
        if not xi:
            raise ValueError("Invalid stream type")
        
        T = 400  # K
        tolerance = 1e-6
        max_iterations = 50
        iterations = 0
        
        while iterations < max_iterations:
            f_T = f(T, xi, P)
            df_T = df(T, xi)
            
            T_new = T - f_T/df_T
            
            if abs(T_new - T) < tolerance:
                PSi = self.Compute_Antoine(T_new)
                return T_new, PSi
            
            T = T_new
            iterations += 1
        
        raise RuntimeError("Newton-Raphson method did not converge")
        
    def Compute_Dew_T(self, stream, P):
            left, right = 200, 600  # K
            solution, iterations = -1, 0
            
            compositions = {'feed': self.xif, 'distillate': self.xid, 'bottom': self.xib}
            xi = compositions.get(stream)
            if not xi:
                raise ValueError("Invalid stream type")
            
            while abs(solution) > 0.001 and iterations < 100:
                T = (left + right) / 2
                PSi = self.Compute_Antoine(T)
                solution = P * sum(x / ps for x, ps in zip(xi, PSi)) - 1
                
                if solution > 0:
                    left = T
                else:
                    right = T
                iterations += 1
            
            return T, PSi
        
    def calculate_relative_volatilities(self, PSi):
        idx_HK = self.components.index("1-nonene")  
        return [ps / PSi[idx_HK] for ps in PSi]  # Ensure HK is reference
    
    def calculate_avg_alpha_LK_HK(self, dist_PSi, bottom_PSi):
        # Calculate relative volatilities for distillate and bottoms
        alpha_dist = self.calculate_relative_volatilities(dist_PSi)
        alpha_bottom = self.calculate_relative_volatilities(bottom_PSi)
        
        # Get indices for light key and heavy key
        idx_LK = self.components.index("1-octene")  # Light Key
        idx_HK = self.components.index("1-nonene")  # Heavy Key
        
        # Calculate geometric mean using equation 6
        # Get the LK-HK relative volatility by dividing LK by HK volatility
        alpha_dist_LK_HK = alpha_dist[idx_LK] / alpha_dist[idx_HK]
        alpha_bottom_LK_HK = alpha_bottom[idx_LK] / alpha_bottom[idx_HK]
        
        avg_alpha_LK_HK = np.sqrt(alpha_dist_LK_HK * alpha_bottom_LK_HK)
        
        return avg_alpha_LK_HK
    
    def Calculate_Fenske(self, avg_alpha_LK_HK):
        try:
            idx_LK = self.components.index("1-octene")  # Light Key (LK)
            idx_HK = self.components.index("1-nonene") 
            # Add absolute value and add 1 to ensure positive stages
            return abs(np.log((self.xid[idx_LK] / self.xid[idx_HK])*(self.xib[idx_HK] / self.xib[idx_LK])) / np.log(avg_alpha_LK_HK))
        except Exception as e:
            print(f"Fenske calculation error: {e}")
            return None    

    def Compute_Underwood1(self, alpha, q=1):
        idx_LK = self.components.index("1-octene")  # Light Key (LK)
        idx_HK = self.components.index("1-nonene")  # Heavy Key (HK)
        left, right = alpha[idx_HK], alpha[idx_LK]  # α_HK < phi < α_LK 
        solution, iterations = -1, 0
        
        while abs(solution) > 0.001 and iterations < 100:
            phi = (left + right) / 2
            sum_terms = sum((self.xif[i] * alpha[i]) / (alpha[i] - phi) for i in range(len(self.xif)))
            solution = sum_terms - (1 - q)
            
            if solution > 0:
                right = phi
            else:
                left = phi
            iterations += 1
        
        return phi 

    def Compute_Underwood2(self, alpha, phi):
            sum_terms = sum((self.xid[i] * alpha[i]) / (alpha[i] - phi) for i in range(len(self.xid)))
            R_min = sum_terms - 1
            return R_min, 1.3 * R_min

    def Compute_Gilliland(self, R_min, R_op, N_min):
            X = (R_op - R_min) / (R_op + 1)
            Y = 1 - np.exp(((1 + 54.4 * X) / (11 + 117.2 * X)) * ((X - 1) / (X ** 0.5)))
            return (Y + N_min) / (1 - Y)

    def Compute_Kirkbride(self, N):
            ratio = ((self.xif[2] / self.xif[1]) * ((self.xib[1] / self.xid[2]) ** 2) * (self.B / self.D)) ** 0.206
            N_rect = ratio * N / (1 + ratio)
            N_strip = N / (1 + ratio)
            return N_rect, N_strip
        
    def calculate_condenser_duty(self,R_op):
            delta_H_VL = sum(x * lh for x, lh in zip(self.xid, self.lambda_id))
            Q_con = self.D * (1 + R_op) * delta_H_VL
            return Q_con / 1e6
        
    def calculate_reboiler_duty(self,V_bar):
            delta_H_VB = sum(x * lh for x, lh in zip(self.xib, self.lambda_ib))
            Q_reb = V_bar * delta_H_VB
            return Q_reb/1e6
        
    def calculate_overall_efficiency(self, dist_dew_T, bottom_bubble_T, avg_alpha_LK_HK,N):
        T_avg = (dist_dew_T + bottom_bubble_T) / 2
        # Calculate numerator and denominator for average viscosity
        numerator = sum(xif_i * mui_i * (MWi_i ** 0.5) 
                    for xif_i, mui_i, MWi_i in zip(self.xif, self.mu_i.values(), self.MW_i))
        denominator = sum(xif_i * (MWi_i ** 0.5) 
                      for xif_i, MWi_i in zip(self.xif, self.MW_i))
        
        if denominator == 0:
            raise ValueError("Denominator is zero in viscosity calculation")
            
        mubarL = numerator / denominator
        mu_barL_cP = mubarL * 1000  # Convert to centipoise
        
        # Calculate overall efficiency using O'Connell correlation
        E0 = 50.3 * (mu_barL_cP * avg_alpha_LK_HK) ** (-0.226)
        
        N_actual = N/(E0/100)
        
        return E0, T_avg,N_actual
    
    def calculate_condenser_area(self, dist_dew_T, dist_bubble_T, Qc):
            cond_in_T = 25  ## in celsius
            cond_out_T = 90 ## in celsius
            dist_dew_T = dist_dew_T - 273.15
            dist_bubble_T = dist_bubble_T - 273.15
            cond_heat_transfer_coeff = 850 ## W m-2 C-1
            LMTD_condenser = (((dist_dew_T-cond_out_T)-(dist_bubble_T-cond_in_T)))/np.log(((dist_dew_T-cond_out_T)/(dist_bubble_T-cond_in_T)))
            condenser_area = (Qc * 1e6) / (LMTD_condenser * cond_heat_transfer_coeff)
            return condenser_area
        
    def calculate_reboiler_area(self, bottom_dew_T, bottom_bubble_T, Qr):
            reb_in_T = 212.38 ## in celsius
            reb_out_T = 212.38 ## in celsius
            bottom_dew_T = bottom_dew_T - 273.15
            bottom_bubble_T = bottom_bubble_T -273.15
            reb_heat_transfer_coeff = 750 ## W m-2 C-1
            LMTD_reboiler = (( reb_in_T - bottom_bubble_T) - ( reb_out_T - bottom_dew_T ))/(np.log((bottom_bubble_T - reb_in_T )/(bottom_dew_T - reb_out_T )))
            reboiler_area = (Qr * 1e6 / (LMTD_reboiler * reb_heat_transfer_coeff))
            return reboiler_area
    
    def calculate_rectifying_diameter(self, dist_dew_T, P, R_op):

        # 1. Average molecular weights calculation
        M_vapor = sum(yi * mi for yi, mi in zip(self.xid, self.MW_i)) # g/mol
        
        # 2. Vapor molar density calculation [mol/m³]
        R = 8.3145  # m³·Pa/(mol·K)
        P_bar = P * 101325  # Convert atm to Pa 
        rho_V_molar = P_bar / (R * dist_dew_T)  # mol/m³
        
        # Given liquid molar density [mol/m³]
        rho_L_molar = 6348 
        
        # 3. Convert molar densities to mass densities [kg/m³]
        rho_V = rho_V_molar * (M_vapor/1000)  # kg/m³
        rho_L = rho_L_molar * (M_vapor/1000)  # kg/m³
        
        # 4. Mass flow rate calculations
        L = self.D * R_op  # mol/s
        V = L + self.D     # mol/s
        
        # Convert mol/s to kg/s
        mass_L = L * (M_vapor/1000)
        mass_V = V * (M_vapor/1000)
        
        # 5. FLV calculation
        if mass_V == 0 or rho_L == 0 or rho_V == 0:
            raise ValueError("Zero value detected in FLV calculation")
        
        FLV = (mass_L/mass_V) * np.sqrt(rho_V/rho_L)
        
        # 6. Flooding velocity calculation
        L_surface_tension_dist = 12.49 # dyne/cm
        K1 = 0.093333
        
        uf = K1 * np.sqrt((rho_L - rho_V)/(rho_V)) * ((L_surface_tension_dist/20)**0.2)
        uv = 0.85 * uf  # Operating velocity
        
        # 7. Column diameter calculation
        A = (mass_V)/(rho_V * uv * (1 - 0.06))  # Cross-sectional area [m²]
        Dc = np.sqrt((4 * A)/np.pi)  # Diameter [m]
                
        return FLV, Dc, rho_L, rho_V,mass_L,mass_V

    def calculate_stripping_diameter(self, bottom_bubble_T, P, R_op):
      
        # 1. Average molecular weights calculation
        M_vapor = sum(yi * mi for yi, mi in zip(self.xib, self.MW_i))  # g/mol
        
        # 2. Vapor molar density calculation [mol/m³]
        R = 8.3145   
        P_bar = P * 101325  # Convert atm to Pa 
        rho_V_molar_strip = P_bar / (R * bottom_bubble_T)  # mol/m³
        
        # Given liquid molar density [mol/m³]
        rho_L_molar_strip = 5280  
        
        # 3. Convert molar densities to mass densities [kg/m³]
        rho_V_strip = rho_V_molar_strip * (M_vapor/1000)  # kg/m³
        rho_L_strip = rho_L_molar_strip *(M_vapor/1000)  # kg/m³
        
        # 4. Mass flow rate calculations
        L_bar = self.D * R_op + self.F  # mol/s
        V_bar = L_bar - self.B     # mol/s
        
        # Convert kmol/s to kg/s
        mass_L_bar = L_bar * (M_vapor/1000)
        mass_V_bar = V_bar * (M_vapor/1000)
        
        # 5. FLV calculation
        FLV_strip = (mass_L_bar/mass_V_bar) * np.sqrt(rho_V_strip/rho_L_strip)
        
        # 6. Flooding velocity calculation
        L_surface_tension_bot = 11.14  # dyne/cm
        K1 = 0.09333
        
        uf_strip = K1 * np.sqrt((rho_L_strip - rho_V_strip)/rho_V_strip) * ((L_surface_tension_bot/20)**0.2)
        uv_strip = 0.85 * uf_strip
        
        # 7. Column diameter calculation
        A_strip = (mass_V_bar)/(rho_V_strip * uv_strip * (1 - 0.06))
        Dc_strip = np.sqrt((4 * A_strip)/np.pi)
                
        return FLV_strip, Dc_strip, rho_L_strip, rho_V_strip
    
    def calculate_column_height(self):
        tray_spacing = 0.55 
        column_height =  (47 * tray_spacing) + (7*tray_spacing)
        return column_height
     
    def run_design_calculations(self, P_atm=1, q=1):
        print("\nSelected components:", self.components)
        P = 1.01325 * P_atm
        # Compute bubble and dew points
        feed_bubble_T, feed_PSi = self.Compute_Bubble_T('feed', P)
        dist_bubble_T, dist_PSi = self.Compute_Bubble_T('distillate', P)
        bottom_bubble_T, bottom_PSi = self.Compute_Bubble_T('bottom', P)
            
        feed_dew_T, feed_dew_PSi = self.Compute_Dew_T('feed', P)
        dist_dew_T, dist_dew_PSi = self.Compute_Dew_T('distillate', P)
        bottom_dew_T, bottom_dew_PSi = self.Compute_Dew_T('bottom', P)
        
        # Compute relative volatilities at bubble and dew points
        alpha_bubble = self.calculate_relative_volatilities(feed_PSi)
        alpha_dew = self.calculate_relative_volatilities(feed_dew_PSi)
                
        avg_alpha_LK_HK = self.calculate_avg_alpha_LK_HK(dist_PSi, bottom_PSi)
        print(f"\nAverage Alpha LK-HK: {avg_alpha_LK_HK}")    
        
        # Compute minimum stages using Fenske with average alpha
        N_min = self.Calculate_Fenske(avg_alpha_LK_HK) - 1
        
        # Compute other parameters (Underwood, Gilliland, etc.)
        phi = self.Compute_Underwood1(alpha_bubble, q)
        R_min, R_op = self.Compute_Underwood2(alpha_bubble, phi)
        N = self.Compute_Gilliland(R_min, R_op, N_min)
        N_rect, N_strip = self.Compute_Kirkbride(N)
        
        # Compute condenser and reboiler duties
        Qc = self.calculate_condenser_duty(R_op)
        L = R_op * self.D
        L_bar = L + self.F  
        V_bar = L_bar - self.B
        Qr = self.calculate_reboiler_duty(V_bar)
        E0,T_avg,N_actual = self.calculate_overall_efficiency(dist_dew_T, bottom_bubble_T, avg_alpha_LK_HK,N)
        condenser_area = self.calculate_condenser_area(dist_dew_T, dist_bubble_T, Qc)
        reboiler_area = self.calculate_reboiler_area(bottom_dew_T, bottom_bubble_T, Qr)
        FLV,RDc,rho_L,rho_V,mass_L,mass_V = self.calculate_rectifying_diameter(dist_dew_T, P,R_op)
        FLV_strip,SDc,rho_L_strip,rho_V_strip = self.calculate_stripping_diameter(bottom_bubble_T, P, R_op)
        column_height = self.calculate_column_height()
                
        # Print results 
        print("\n" + "-"*10 + " PHASE EQUILIBRIUM ANALYSIS " + "-"*10)
        print("\nFeed:")
        print(f"Bubble temperature: {np.round(feed_bubble_T, 2)} K ({np.round(feed_bubble_T - 273.15, 2)} °C)")
        print(f"Dew temperature: {np.round(feed_dew_T, 2)} K ({np.round(feed_dew_T - 273.15, 2)} °C)")
        
        print("\nDistillate:")
        print(f"Bubble temperature: {np.round(dist_bubble_T, 2)} K ({np.round(dist_bubble_T - 273.15, 2)} °C)")
        print(f"Dew temperature: {np.round(dist_dew_T, 2)} K ({np.round(dist_dew_T - 273.15, 2)} °C)")
        
        print("\nBottom:")
        print(f"Bubble temperature: {np.round(bottom_bubble_T, 2)} K ({np.round(bottom_bubble_T - 273.15, 2)} °C)")
        print(f"Dew temperature: {np.round(bottom_dew_T, 2)} K ({np.round(bottom_dew_T - 273.15, 2)} °C)")
        
        print("\n" + "-"*10 + " COLUMN DESIGN CALCULATIONS " + "-"*10)
        print(f"\nMinimum stages (N_min): {np.round(N_min, 2)}")
        print(f"Phi (ϕ) : {np.round(phi,7)}")
        print(f"Minimum reflux ratio (R_min): {np.round(R_min, 2)}")
        print(f"Operating reflux ratio (R_op): {np.round(R_op, 2)}")
        print(f"Total stages needed (N): {np.round(N, 2)}")
        print(f"Rectifying stages: {np.round(N_rect, 2)}")
        print(f"Stripping stages: {np.round(N_strip, 2)}")
        
        print("\n" + "-"*10 + " CONDENSER AND REBOILER CONSIDERATIONS " + "-"*10)
        print(f"Condenser Duty (Q_C): {np.round(Qc, 2)} MJ/s (MW)")
        print(f"Reboiler Duty (Q_R): {np.round(Qr, 2)} MJ/s (MW)")
        print(f"Condenser Area (A_cond): {np.round(condenser_area,2)} (m\N{SUPERSCRIPT TWO})")
        print(f"Reboiler Area (A_reb): {np.round(reboiler_area,2)} (m\N{SUPERSCRIPT TWO})")
        
        print("\n" + "-"*10 + " MOLAR FLOWRATES (mol/s) " + "-"*10)
        print(f"Feed (F): {np.round(self.F, 4)}")
        print(f"Distillate (D): {np.round(self.D, 4)}")
        print(f"Bottoms (B): {np.round(self.B, 4)}")
        print(f"Reflux (L): {np.round(L, 4)}")
        print(f"Boilup (V_bar): {np.round(V_bar, 4)}")
        print(f"L_bar: {np.round(L_bar,4)}")

        # Print relative volatilities
        print("\nRelative volatilities (α):")
        print(" ".join(f"{α:5.2f}" for α in alpha_bubble))
        
        # Print Average temperature in column and overall efficiency
        print("\n" + "-" * 10 + " COLUMN EFFICIENCY CALCULATIONS " + "-" * 10)
        print("Overall Column Efficiency: " + str(E0))
        print("Actual Number of stages in the column: "+str(N_actual))
        print("Average Temperature in the Column: " + str(T_avg))
        
        # Print column dimensions
        print("\n" + "-" * 10 + " COLUMN SIZING AND INTERNAL DESIGN " + "-" * 10)
        print(f"FLV: {np.round(FLV,4)}")
        print(f"Rectifying section diameter: {np.round(RDc,4)}")  
        print(f"FLV_strip: {np.round(FLV_strip,4)}")
        print(f"Stripping section diameter: {np.round(SDc,4)}")
        print(f"Column height : {np.round(column_height)+1} (m)")
    
if __name__ == "__main__":
    calculator = FUGK_distillation() 
    results = calculator.run_design_calculations()    
    
        
        
        

        
        


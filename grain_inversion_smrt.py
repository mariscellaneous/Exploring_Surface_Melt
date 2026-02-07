from smrt import make_snowpack, make_model, sensor_list
import numpy as np

class GrainInversionSmrt:
    def __init__(self,sensor='18V',EM='iba',solver='dort',microstructure='exponential'):
        self.sensor = sensor
        self.EM = EM
        self.solver = solver
        self.microstructure = microstructure
        self.sensorV = sensor_list.amsre(self.sensor)
        self.model = make_model(self.EM,self.solver)

    @staticmethod
    def x_intercept(x0, x1, y0, y1):
        m = (y1 - y0) / (x1 - x0)
        b = y1 - m * x1
        return b

    def run_smrt_forward(self,thickness,temperature, grain_guess,density, estimate_tb,liquid_water):
        snowpack2 = make_snowpack(thickness,self.microstructure, density = density, temperature = temperature,corr_length = grain_guess,liquid_water=liquid_water)
        resV = self.model.run(self.sensorV, snowpack2)
        print(resV.TbV())
        difference = resV.TbV()-estimate_tb
        return difference
    
    def calc_bounds(self,init_bounds,estimate_tb,thickness,temperature,density):
        snowpack0 = make_snowpack(thickness,self.microstructure, density = density, temperature = temperature,
                                            corr_length = init_bounds[0])
        snowpack1 = make_snowpack(thickness,self.microstructure, density = density, temperature = temperature,
                                            corr_length = init_bounds[1])

        resV0 = self.model.run(self.sensorV, snowpack0)
        resV1 = self.model.run(self.sensorV, snowpack1)
        difference0 = resV0.TbV()-estimate_tb
        difference1 = resV1.TbV()-estimate_tb

        if ((difference0>0) & (difference1<0)).any()==False:
            print('out of bounds')
        
        offset=np.asarray([difference0,difference1])
        guess = self.x_intercept(difference0,difference1,init_bounds[0],init_bounds[1])
        snowpack2 = make_snowpack(thickness,self.microstructure, density = density, temperature = temperature,
                                            corr_length = guess)    
        resV2 = self.model.run(self.sensorV, snowpack2)
        difference2 = resV2.TbV()-estimate_tb
        return guess,offset

    def min_grain_diff(self,grain_guesses,estimate_tb,thickness,temperature,density):
        difference = grain_guesses.copy()*np.nan
        
        for ind,grain_guess in enumerate(grain_guesses):
            snowpack = make_snowpack(thickness,self.microstructure, density = density, temperature = temperature,
                                        corr_length = grain_guess,liquid_water=0)

            resV = self.model.run(self.sensorV, snowpack)
            difference[ind] = resV.TbV()-estimate_tb   
            
        grain_size_idx = np.argmin(np.abs(difference))
        grain_size = grain_guesses[grain_size_idx]
        final_offset = difference[grain_size_idx]
        
        if ((difference[0]>0) & (difference[-1]<0)).any() == False:
            print('out of bounds2')
        return grain_size,final_offset

    def calc_grain_size(self,init_bounds,estimate_tb,thickness,temperature=280,density=400,uncertainty=150e-6):
        guess,offset = self.calc_bounds(init_bounds,estimate_tb,thickness,temperature,density)
        grain_guesses = np.linspace(guess-uncertainty,guess+uncertainty,10)
        grain_size,final_offset=self.min_grain_diff(grain_guesses,estimate_tb,thickness,temperature,density)
        return grain_size,final_offset

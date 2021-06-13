""" make therm curve """
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os, glob
import random

# parts parameters
parts_list = ["FET", "RC", "MG", "MC"]
fet_param_dict = {"K1":100, "tau1":10, "K2":5, "tau2":100}
rc_param_dict = {"K1":30, "tau1":20, "K2":3, "tau2":4}
mg_param_dict = {"K1":3, "tau1":40, "K2":1, "tau2":10}
mc_param_dict = {"K1":1, "tau1":100, "K2":6, "tau2":9}
parts_parameter = {"FET":fet_param_dict, "RC":rc_param_dict, "MG":mg_param_dict, "MC":mc_param_dict}

# parameters
time_max = 1000
Ts = 1
therm_sweep_list = range(40,121,10)
current_sweep_list = range(20, 101, 20)
mrpm_sweep_list = [160]

def main():
	time_list = range(0, time_max)

	for therm in therm_sweep_list:
		for current in current_sweep_list:
			for mrpm in mrpm_sweep_list:
				current_list = [current for i in range(len(time_list))]
				mrpm_list = [mrpm for i in range(len(time_list))]
				res = {}

				for parts in parts_list:
					K1 = parts_parameter[parts]["K1"]
					tau1 = parts_parameter[parts]["tau1"]
					K2 = parts_parameter[parts]["K2"]
					tau2 = parts_parameter[parts]["tau2"]
					res[parts] = make_curve(time_list, therm, current_list, mrpm_list, Ts, K1, tau1, K2, tau2)
					file_name = str(therm) + "degC_" + str(current) + "A_" + str(mrpm) + "rpm_" + parts
					save_therm_curve("./data/MockData/" + file_name + ".csv", res[parts])


def make_curve(time_list, start_therm, current_list, mrpm_list, Ts, k1, tau1, k2, tau2):
	# init work ram
	T, T1, T2 = [], [], []
	curve_dict = {}

	for t, current in zip(time_list, current_list):
		if len(T1) > 0:
			delta_t1 = k1 * (current * np.sqrt(3)) * Ts
			T1.append( T1[-1] + (delta_t1 - T1[-1])*(1-np.exp(-Ts/tau1)) * 0.001 )
			delta_t2 = k2 * (current * np.sqrt(3)) * Ts
			T2.append( T2[-1] + (delta_t2 - T2[-1])*(1-np.exp(-Ts/tau2)) * 0.001 )
		else:
			T1.append(start_therm)
			T2.append(start_therm)

		# gain
		T.append(T1[-1] + T2[-1] - start_therm)

	curve_dict["time"] = time_list
	curve_dict["current"] = current_list
	curve_dict["mrpm"] = mrpm_list
	curve_dict["T"] = T
	curve_dict["T1"] = T1
	curve_dict["T2"] = T2

	return curve_dict

def save_therm_curve(path, curve_dict):
	os.makedirs(os.path.dirname(path), exist_ok=True)
	df = pd.DataFrame(curve_dict)
	df.to_csv(path)


def make_current_pattern(time_list, pattern_list=range(0, 101, 20), random_flag=False):
	time_len = len(time_list)
	one_ptn_len = int(time_len / len(pattern_list))
	res = []

	if random_flag:
		pattern_list = random.sample(pattern_list, len(pattern_list))

	# make pattern
	for ptn in pattern_list:
		res += [ptn for i in range(one_ptn_len)]

	# mod
	for i in range(time_len - len(res)):
		res.append(res[-1])

	# plt.plot(time_list, res)
	# plt.show()

	return res


def make_pattern_therm(random_flag=True):
	parts = "MC"
	K1 = parts_parameter[parts]["K1"]
	tau1 = parts_parameter[parts]["tau1"]
	K2 = parts_parameter[parts]["K2"]
	tau2 = parts_parameter[parts]["tau2"]

	time_max = 3000
	time_list = [i for i in range(time_max)]
	mrpm_list = [150 for i in range(len(time_list))]
	therm = 40

	ofst = 500
	c_l = np.log10(time_list)
	c_l[0] = 0
	c_l = c_l / np.max(c_l)
	c_l = (1 - c_l)
	# insert
	c_l = np.insert(c_l, 0, [1 for i in range(ofst)])
	c_l = c_l[:len(time_list)]
	c_l = c_l * 100
	# plt.plot(c_l)
	# plt.show()

	# current_pattern_list = range(101, 10, -2)
	current_pattern_list = c_l
	current_list = make_current_pattern(time_list, current_pattern_list, random_flag)

	res = make_curve(time_list, therm, current_list, mrpm_list, Ts, K1, tau1, K2, tau2)
	file_name = str(therm) + "degC_" + "{:.3g}_{:.3g}_".format(min(current_list), max(current_list))\
		+ "A_" + str(150) + "rpm_" + parts
	save_therm_curve("./data/MockDataDcr/" + file_name + ".csv", res)


if __name__ == '__main__':
	make_pattern_therm(random_flag=False)
	# main()
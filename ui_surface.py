from tkinter import messagebox
from tkinter import ttk
from tkinter import *
import tkinter as tk
import option_pricing

# ==================Create instance========================
root = Tk()
root.title("Mini Option Pricer")
root.resizable(0,0)
root.minsize(width=500, height=500)

# ===================Tab Control===========================
tabControl = ttk.Notebook(root)
tab1 = ttk.Frame(tabControl)
tabControl.add(tab1,text="Option")
tab2 = ttk.Frame(tabControl)
tabControl.add(tab2,text="ImpliedVol")
tabControl.pack(expand=1, fill="both")

# =======================Tab1: Option Pricing==============================

# ======================Frame 0: Option type============================
frame0 = ttk.LabelFrame(tab1,text = "Option")
frame0.grid(column = 0,row = 2,padx = 8, pady = 4)
option = tk.StringVar()
option_Chosen = ttk.Combobox(frame0, width=25, textvariable = option)
option_Chosen['values'] = ('European Option',  "American Option", "GeoMetric Asian Option",
                                  "Arithmetic Asian Option",
                                  "GeoMetric Basket Option", "Arithmetic Basket Option")
option_Chosen.grid(column=1, row=0)
option_Chosen.current(0)  # 设置初始显示值，值为元组['values']的下标
option_Chosen.config(state='readonly')  # 设为只读模式


# =====================optoin type=========================

ttk.Label(frame0,text="Option Type").grid(column=2,row=0,sticky='W')
chVarUn1 = tk.StringVar()
check_call = tk.Checkbutton(frame0, text="Call", variable=chVarUn1,onvalue="C")
check_call.deselect()  # Clears (turns off) the checkbutton.
check_call.grid(column=3, row=0, sticky=tk.W,padx=1)


check_put = tk.Checkbutton(frame0, text="Put", variable=chVarUn1,onvalue="P")
check_put.deselect()
check_put.grid(column=4, row=0, sticky=tk.W,padx=1)

for child in frame0.winfo_children():
    child.grid_configure(padx=15,pady=4)

# =====================Frame 1: Parameters========================
frame1 = ttk.LabelFrame(tab1,text = "Parameters")
frame1.grid(column = 0,row = 4,padx = 8, pady = 4)

ttk.Label(frame1,text="Spot Price of asset").grid(column=0,row=0,sticky='W')
spotPrice1 = tk.StringVar()
spotPrice1Entered = ttk.Entry(frame1, width=8, textvariable=spotPrice1)
spotPrice1Entered.grid(column=8, row=0)
spotPrice2 = tk.StringVar()
spotPrice2Entered = ttk.Entry(frame1, width=8, textvariable=spotPrice2)
spotPrice2Entered.grid(column=10, row=0)

ttk.Label(frame1,text="Volatility").grid(column=0,row=2,sticky='W')
vol1 = tk.StringVar()
vol1Entered = ttk.Entry(frame1, width=8, textvariable=vol1)
vol1Entered.grid(column=8, row=2)
vol2 = tk.StringVar()
vol2Entered = ttk.Entry(frame1, width=8, textvariable=vol2)
vol2Entered.grid(column=10, row=2)

ttk.Label(frame1,text="Time to Maturity (years)").grid(column=0,row=4,sticky='W')
time = tk.StringVar()
timeEntered = ttk.Entry(frame1, width=8, textvariable=time)
timeEntered.grid(column=8, row=4)

ttk.Label(frame1,text="Strike Price").grid(column=0,row=6,sticky='W')
strike = tk.StringVar()
strikeEntered = ttk.Entry(frame1, width=8, textvariable=strike)
strikeEntered.grid(column=8, row=6)

ttk.Label(frame1,text="Risk-Free rate").grid(column=0,row=8,sticky='W')
rate = tk.StringVar()
rateEntered = ttk.Entry(frame1, width=8, textvariable=rate)
rateEntered.grid(column=8, row=8)

ttk.Label(frame1,text="Repo rate").grid(column=0,row=10,sticky='W')
repo = tk.StringVar()
repoEntered = ttk.Entry(frame1, width=8, textvariable=repo)
repoEntered.grid(column=8, row=10)

ttk.Label(frame1,text="Correlation").grid(column=0,row=12,sticky='W')
correlation = tk.StringVar()
correlationEntered = ttk.Entry(frame1, width=8, textvariable=correlation)
correlationEntered.grid(column=8, row=12)


ttk.Label(frame1,text="Steps").grid(column=25,row=0,sticky='W')
steps = tk.StringVar()
stepsEntered = ttk.Entry(frame1, width=8, textvariable=steps)
stepsEntered.grid(column=27, row=0)

ttk.Label(frame1,text="Paths").grid(column=25,row=2,sticky='W')
paths = tk.StringVar()
pathsEntered = ttk.Entry(frame1, width=8, textvariable=paths)
pathsEntered.grid(column=27, row=2)

ttk.Label(frame1,text="Control Variate").grid(column=25,row=4,sticky='W')

chVarUn2 = tk.StringVar()
check_CV = tk.Checkbutton(frame1, text="CV", variable=chVarUn2,onvalue="Yes")
check_CV.deselect()  # Clears (turns off) the checkbutton.
check_CV.grid(column=27, row=4, sticky=tk.W)


check_NCV = tk.Checkbutton(frame1, text="Non-CV", variable=chVarUn2,onvalue="NO")
check_NCV.deselect()
check_NCV.grid(column=29, row=4, sticky=tk.W)


# =======================Calculate Option==========================
def calculate_option():

    try:
        select = option.get()
        S1 = float(spotPrice1.get())
        T = float(time.get())
        r = float(rate.get())
        K = float(strike.get())
        type = chVarUn1.get()
        cv = chVarUn2.get()
        resultPrice = 0.0
        resultPrice1 = 0.0
        resultPrice2 = 0.0
        judge = False

        if type =='0':
            raise ValueError()
            # messagebox.showinfo("Please check your parameters", "Please check and input all parameters")

        # option_Chosen['values'] = ('European Option', "American Option", "GeoMetric Asian Option",
        #                            "Arithmetic Asian Option",
        #                            "GeoMetric basket Option", "Arithmetic basket Option")

        if select == option_Chosen['values'][0]:  # "European Option":

            sigma1 = float(vol1.get())
            q = float(repo.get())

            if type == "C":
                #price_call(S, K, r, q, sigma, tau)
                resultPrice = option_pricing.price_call(S1,K,r,q,sigma1,T)
            elif type == "P":
                # price_put(S, K, r, q, sigma, tau)
                resultPrice = option_pricing.price_put(S1,K,r,q,sigma1,T)
            else:
                resultPrice = 0.0


        elif select == option_Chosen['values'][1]:  # "American Option":

            n = int(steps.get())
            sigma1 = float(vol1.get())

            resultPrice = option_pricing.Bionomial_tree(S1,K, r,sigma1, T, n, type)


        elif select == option_Chosen['values'][2]:  # "GeoMetric Asian Option":
            n = int(steps.get())
            sigma1 = float(vol1.get())
            #geometric_Asian_option(S, sigma, r, T, K, n, option_type)
            resultPrice = option_pricing.geometric_Asian_option(S1, sigma1, r, T, K, n, type)


        elif select == option_Chosen['values'][3]:  # "Arithmetic Asian Option":
            if cv == '0':
                # messagebox.showinfo("Please check your parameters", "Please check and input all parameters")
                raise ValueError()
            judge = True
            n = int(steps.get())
            path = int(paths.get())
            sigma1 = float(vol1.get())
            cv = chVarUn2.get()


            # ArithmeticAsianOptionMC(T, K, n, S, r, sigma, option_type, NumPath, cv = 'NO')
            resultPrice1,resultPrice2  = option_pricing.ArithmeticAsianOptionMC(T, K, n, S1, r, sigma1, type, path, cv)
            # print("resultPrice: {0}     interval:  {1}".format(resultPrice1,resultPrice2))


        elif select == option_Chosen['values'][4]:  # 'GeoMetric basket option':
            S2 = float(spotPrice2.get())
            sigma1 = float(vol1.get())
            sigma2 = float(vol2.get())
            corr = float(correlation.get())
            # geometric_basket_option(S1, S2, sigma1, sigma2, r, T, K, rou, option_type)
            resultPrice = option_pricing.geometric_basket_option(S1, S2, sigma1, sigma2, r, T, K, corr, type)


        elif select == option_Chosen['values'][5]:  # 'Arithmetic basket option':
            if cv == '0':
                # messagebox.showinfo("Please check your parameters", "Please check and input all parameters")
                raise ValueError()
            judge = True
            S2 = float(spotPrice2.get())
            sigma1 = float(vol1.get())
            sigma2 = float(vol2.get())
            corr = float(correlation.get())
            path = int(paths.get())
            cv = chVarUn2.get()

            # ArithmeticBasketOptionMC(T, K, S1, S2, r, sigma1, sigma2, rou, option_type,NumPath, cv='NO')
            resultPrice1,resultPrice2 = option_pricing.ArithmeticBasketOptionMC(T, K, S1, S2, r, sigma1, sigma2, corr, type, path, cv)
            # print("resultPrice: {0}     interval:  {1}".format(resultPrice1, resultPrice2))
            # print(ArithmeticBasketOptionMC(T, 100, 100, 100, r, 0.3, 0.3, 0.5, 'PUT', 100000))
            # print(ArithmeticBasketOptionMC(T, 100, 100, 100, r, 0.3, 0.3, 0.5, 'PUT', 100000, 'YES'))
        if judge == True:
            # textEntered['text'] = "Price: %.5f \nInterval: %.5f %.5f" % (resultPrice1, resultPrice2[0], resultPrice2[1])
            textEntered['text'] = "Option Price: {:.5f} \nInterval: [{:.5f} , {:.5f}]".format(resultPrice1,
                                                                                              resultPrice2[0],
                                                                                              resultPrice2[1])
        else:
            textEntered['text'] = "Option Price: {:.5f}".format(resultPrice)
    except ValueError as e:
        # print(e)
        messagebox.showinfo("Please check your parameters", "Please check all the inputs")


result_Btn = ttk.Button(frame1,text="Calculate",width=10,command=calculate_option)
result_Btn.grid(column=25,row=8,rowspan=10,ipady=7)


for child in frame1.winfo_children():
    child.grid_configure(padx=8,pady=6)

# ===================frame 2: Results=======================

frame2 = ttk.LabelFrame(tab1,text = "Results",width=500,height=500)
frame2.grid(column = 0,row = 25,padx = 8, pady = 5)


textEntered = tk.Label(frame2, width=60,height=5)
textEntered.grid(column=0, row=8)



# =======================Tab 2: Implied Vol=====================


framePara = ttk.Frame(tab2)
framePara.pack(side=TOP)
frame3 = ttk.LabelFrame(framePara,text = "Parameters",width=50,height=50)
frame3.grid(column = 0,row = 4,padx = 4, pady = 4)
# frame3.place(in_=tab2, anchor="n", relx=.5,rely=.05)

ttk.Label(frame3,text="Spot Price of asset").grid(column=2,row=0,sticky='W')
Vol_S = tk.StringVar()
Vol_SEntered = ttk.Entry(frame3, width=8, textvariable=Vol_S)
Vol_SEntered.grid(column=5, row=0)

ttk.Label(frame3,text="Strike Price").grid(column=2,row=2,sticky='W')
Vol_K = tk.StringVar()
Vol_KEntered = ttk.Entry(frame3, width=8, textvariable=Vol_K)
Vol_KEntered.grid(column=5, row=2)

ttk.Label(frame3,text="Time to Maturity").grid(column=2,row=4,sticky='W')
Vol_T = tk.StringVar()
Vol_TEntered = ttk.Entry(frame3, width=8, textvariable=Vol_T)
Vol_TEntered.grid(column=5, row=4)

ttk.Label(frame3,text="Risk-Free rate").grid(column=2,row=6,sticky='W')
Vol_R = tk.StringVar()
Vol_REntered = ttk.Entry(frame3, width=8, textvariable=Vol_R)
Vol_REntered.grid(column=5, row=6)

ttk.Label(frame3,text="Repo Rate").grid(column=2,row=8,sticky='W')
Vol_rho = tk.StringVar()
Vol_rhoEntered = ttk.Entry(frame3, width=8, textvariable=Vol_rho)
Vol_rhoEntered.grid(column=5, row=8)

ttk.Label(frame3,text="Premium").grid(column=2,row=10,sticky='W')
Vol_P = tk.StringVar()
Vol_PEntered = ttk.Entry(frame3, width=8, textvariable=Vol_P)
Vol_PEntered.grid(column=5, row=10)

ttk.Label(frame3,text="Option Type").grid(column=2,row=12,sticky='W')
vol_type = tk.StringVar()
vol_call = tk.Checkbutton(frame3, text="Call", variable=vol_type,onvalue="C")
vol_call.deselect()  # Clears (turns off) the checkbutton.
vol_call.grid(column=4, row=12, sticky=tk.W,padx=1)

vol_put = tk.Checkbutton(frame3, text="Put", variable=vol_type,onvalue="P")
vol_put.deselect()
vol_put.grid(column=5, row=12, sticky=tk.W,padx=1)


def clickMe2():
    #result_Btn.configure(text='calculating ' )
    Vol_Btn.configure(state='disabled')  # Disable the Button Widget

def reset_option():
    for field in fields:
        field.delete(0, END)
    textEntered['text'] = ""
    check_call.deselect()
    check_put.deselect()
    check_NCV.deselect()
    check_CV.deselect()
fields = spotPrice1Entered,spotPrice2Entered,vol1Entered,vol2Entered,timeEntered, rateEntered, strikeEntered, correlationEntered,stepsEntered, pathsEntered, repoEntered

reset_Btn = ttk.Button(frame1,text="Reset",width=10,command=reset_option)
reset_Btn.grid(column=27,row=8,rowspan=10,ipady=7)

def calculate_vol():
    resultPrice = 0
    try:
        S_vol = float(Vol_S.get())
        T_vol = float(Vol_T.get())
        rho_vol = float(Vol_rho.get())
        r_vol = float(Vol_R.get())
        K_vol = float(Vol_K.get())
        type = vol_type.get()
        premium = float(Vol_P.get())

        if type == '0':
            messagebox.showinfo("Please check your parameters", "Please check all the inputs")
        if type == "C":
            # v_call(S, r, q, tau, K, C_true)
            resultPrice = option_pricing.v_call(S_vol,r_vol,rho_vol,T_vol,K_vol,premium)
        elif type == "P":
            # v_put(S, r, q, tau, K, P_true)
            resultPrice = option_pricing.v_put(S_vol,r_vol,rho_vol, T_vol, K_vol,premium)
        else:
            resultPrice = 0.0
    except ValueError as e:
        # print(e)
        messagebox.showinfo("Please check your parameters", "Please check and input all parameters")
    textEntered2['text'] = "%.5f" % resultPrice

Vol_Btn = ttk.Button(frame3,text="Calculate",width=10,command=calculate_vol)
Vol_Btn.grid(column=3,row=16,rowspan=10,ipady=5)

frameImpliedVol = ttk.Frame(tab2)
frameImpliedVol.pack(side=TOP)
frame4 = ttk.LabelFrame(frameImpliedVol,text = "Implied Vol",width=500,height=500)
frame4.grid(column = 0,row = 25,padx = 8, pady = 5)


textEntered2 = tk.Label(frame4, width=60,height=5)
textEntered2.grid(column=0, row=8)

def reset_vol():
    for field in fields_vol:
        field.delete(0, END)
    vol_call.deselect()
    vol_put.deselect()
    textEntered2["text"]=""
fields_vol = Vol_SEntered,Vol_KEntered, Vol_PEntered, Vol_REntered, Vol_rhoEntered,Vol_TEntered,repoEntered
reset_Btn = ttk.Button(frame3,text="Reset",width=10,command=reset_vol)
reset_Btn.grid(column=4,row=16,rowspan=10,ipady=5)

for child in frame3.winfo_children():
    child.grid_configure(padx=8,pady=6)


# ====================end Layout=====================


def selected(*args):
    select = option_Chosen.get()
    if select == option_Chosen['values'][0]: #'European call/put option':
        spotPrice1Entered.configure(state='normal')
        spotPrice2Entered.configure(state='disabled')
        vol1Entered.configure(state='normal')
        vol2Entered.configure(state='disabled')
        timeEntered.configure(state='normal')
        rateEntered.configure(state='normal')
        strikeEntered.configure(state='normal')
        correlationEntered.configure(state='disabled')
        stepsEntered.configure(state='disabled')
        pathsEntered.configure(state='disabled')
        check_NCV.configure(state='disabled')
        check_CV.configure(state='disabled')
        repoEntered.configure(state='normal')
        check_call.configure(state='normal')
        check_put.configure(state='normal')
    elif select == option_Chosen['values'][1]: #'American call/put option':
        spotPrice1Entered.configure(state='normal')
        spotPrice2Entered.configure(state='disabled')
        vol1Entered.configure(state='normal')
        vol2Entered.configure(state='disabled')
        timeEntered.configure(state='normal')
        rateEntered.configure(state='normal')
        strikeEntered.configure(state='normal')
        correlationEntered.configure(state='disabled')
        stepsEntered.configure(state='normal')
        pathsEntered.configure(state='disabled')
        check_NCV.configure(state='disabled')
        check_CV.configure(state='disabled')
        repoEntered.configure(state='disabled')
        check_call.configure(state='normal')
        check_put.configure(state='normal')
    elif select == option_Chosen['values'][2]: #'GeoMetric Asian option':
        spotPrice1Entered.configure(state='normal')
        spotPrice2Entered.configure(state='disabled')
        vol1Entered.configure(state='normal')
        vol2Entered.configure(state='disabled')
        timeEntered.configure(state='normal')
        rateEntered.configure(state='normal')
        strikeEntered.configure(state='normal')
        correlationEntered.configure(state='disabled')
        stepsEntered.configure(state='normal')
        pathsEntered.configure(state='disabled')
        check_NCV.configure(state='disabled')
        check_CV.configure(state='disabled')
        repoEntered.configure(state='disabled')
        check_call.configure(state='normal')
        check_put.configure(state='normal')
    elif select == option_Chosen['values'][3]: #'Arithmetic Asian option':
        spotPrice1Entered.configure(state='normal')
        spotPrice2Entered.configure(state='disabled')
        vol1Entered.configure(state='normal')
        vol2Entered.configure(state='disabled')
        timeEntered.configure(state='normal')
        rateEntered.configure(state='normal')
        strikeEntered.configure(state='normal')
        correlationEntered.configure(state='disabled')
        stepsEntered.configure(state='normal')
        pathsEntered.configure(state='normal')
        check_NCV.configure(state='normal')
        check_CV.configure(state='normal')
        repoEntered.configure(state='disabled')
        check_call.configure(state='normal')
        check_put.configure(state='normal')
    elif select == option_Chosen['values'][4]: #'GeoMetric basket option':
        spotPrice1Entered.configure(state='normal')
        spotPrice2Entered.configure(state='normal')
        vol1Entered.configure(state='normal')
        vol2Entered.configure(state='normal')
        timeEntered.configure(state='normal')
        rateEntered.configure(state='normal')
        strikeEntered.configure(state='normal')
        correlationEntered.configure(state='normal')
        stepsEntered.configure(state='disabled')
        pathsEntered.configure(state='disabled')
        check_NCV.configure(state='disabled')
        check_CV.configure(state='disabled')
        repoEntered.configure(state='disabled')
        check_call.configure(state='normal')
        check_put.configure(state='normal')
    elif select == option_Chosen['values'][5]: #'Arithmetic basket option':
        spotPrice1Entered.configure(state='normal')
        spotPrice2Entered.configure(state='normal')
        vol1Entered.configure(state='normal')
        vol2Entered.configure(state='normal')
        timeEntered.configure(state='normal')
        rateEntered.configure(state='normal')
        strikeEntered.configure(state='normal')
        correlationEntered.configure(state='normal')
        stepsEntered.configure(state='disabled')
        pathsEntered.configure(state='normal')
        check_NCV.configure(state='normal')
        check_CV.configure(state='normal')
        repoEntered.configure(state='disabled')
        check_call.configure(state='normal')
        check_put.configure(state='normal')
option_Chosen.bind("<<ComboboxSelected>>", selected)
selected()

mainloop()
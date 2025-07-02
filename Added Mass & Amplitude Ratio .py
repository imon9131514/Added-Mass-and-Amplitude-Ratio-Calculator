import numpy as np
from scipy.interpolate import RegularGridInterpolator

# Define the equations for Added Mass Coefficient (C) based on B and T values
equations_c = {
    1.0: {
        0.4: lambda x: 25.4319*x**8 - 167.803*x**7 + 462.976*x**6 - 692.285*x**5 + 606.986*x**4 - 313.851*x**3 + 89.7265*x**2 - 11.1022*x + 1.58697,
        0.8: lambda x: 8.72678*x**8 - 59.5552*x**7 + 171.825*x**6 - 273.375*x**5 + 262.495*x**4 - 156.118*x**3 + 55.7776*x**2 - 10.004*x + 1.72948,
        1.2: lambda x: 1.9763*x**6 - 11.4921*x**5 + 27.3004*x**4 - 33.7212*x**3 + 22.3441*x**2 - 6.81887*x + 1.74686,
        1.6: lambda x: 5.80924*x**8 - 36.9516*x**7 + 97.6292*x**6 - 140.055*x**5 + 121.659*x**4 - 69.8161*x**3 + 28.7127*x**2 - 7.63228*x + 1.85593,
        2.0: lambda x: 8.50462*x**8 - 54.9729*x**7 + 148.759*x**6 - 220.06*x**5 + 196.304*x**4 - 111.415*x**3 + 41.9802*x**2 - 10.1142*x + 2.08731,
        2.4: lambda x: 2.40044*x**8 - 18.8776*x**7 + 62.9977*x**6 - 116.404*x**5 + 130.579*x**4 - 92.151*x**3 + 41.151*x**2 - 11.0297*x + 2.25831,
        2.8: lambda x: 9.62464*x**8 - 67.3263*x**7 + 198.551*x**6 - 321.421*x**5 + 312.362*x**4 - 188.281*x**3 + 70.9139*x**2 - 16.1537*x + 2.66333,
        3.6: lambda x: 3.68045*x**8 - 27.7851*x**7 + 89.6315*x**6 - 161.252*x**5 + 177.029*x**4 - 122.079*x**3 + 52.7368*x**2 - 13.7042*x + 2.59361,
        4.4: lambda x: -0.0247978*x**8 - 0.0510849*x**7 + 1.8129*x**6 - 8.62986*x**5 + 19.295*x**4 - 24.1605*x**3 + 17.7871*x**2 - 7.40725*x + 2.20208,
    },
    0.9: {
        0.4: lambda x: (1.5234*x**5 + 4.0876*x**4 + 10.4239*x**3 -1.9445*x**2 + 0.5609*x -0.0101) / (x**5 -1.6302*x**4 - 5.6165*x**3 + 2.9821*x**2 + 0.3513*x -0.0083) if x<=0.138 else 1.1138*x**6 - 5.4856*x**5 + 10.206*x**4 - 8.4133*x**3 + 1.9818*x**2 + 1.3367*x + 0.4896,
        0.8: lambda x: 4.492*x**6 - 24.037*x**5 + 51.672*x**4 - 56.83*x**3 + 33.164*x**2 - 8.9968*x + 1.5999,
        1.2: lambda x: 2.0991*x**6 - 12.429*x**5 + 29.914*x**4 - 37.67*x**3 + 26.214*x**2 - 9.1688*x + 1.9691,
        1.6: lambda x: 2776.3*x**6 - 5152.4*x**5 + 3885.2*x**4 - 1526.3*x**3 + 333.69*x**2 - 40.175*x + 3.0001 if x<=0.45863 else 0.4379*x**4 - 2.0727*x**3 + 3.4093*x**2 - 2.0179*x + 1.0926, 
        2.0: lambda x: 0.349*x**6 - 3.1328*x**5 + 10.381*x**4 - 17.001*x**3 + 14.9*x**2 - 6.5617*x + 1.8481,
        2.4: lambda x: 0.7098*x**4 - 3.1786*x**3 + 5.3947*x**2 - 3.9685*x + 1.8,
        2.8: lambda x: 0.6954*x**4 - 3.086*x**3 + 5.1978*x**2 - 3.8986*x + 1.846,
        3.6: lambda x: 1.1809*x**4 - 4.8807*x**3 + 7.3177*x**2 - 4.6652*x + 1.8015,
        4.4: lambda x: 0.3192*x**6 - 2.4602*x**5 + 7.7761*x**4 - 12.977*x**3 + 12.052*x**2 - 5.8356*x + 1.8547,
    },
    0.8: {
        0.4: lambda x: 10.812*x**6 - 53.738*x**5 + 104.95*x**4 - 101.41*x**3 + 49.498*x**2 - 10.415*x + 1.2262,
        0.8: lambda x: 6.1529*x**6 - 31.075*x**5 + 62.671*x**4 - 64.519*x**3 + 35.654*x**2 - 9.612*x + 1.5017,
        1.2: lambda x: 0.8743*x**6 - 5.8042*x**5 + 15.816*x**4 - 22.619*x**3 + 17.835*x**2 - 7.0515*x + 1.6204,
        1.6: lambda x: 0.0079*x**6 - 1.3853*x**5 + 6.6705*x**4 - 12.804*x**3 + 12.291*x**2 - 5.8175*x + 1.6589,
        2.0: lambda x: 2.7008*x**6 - 13.743*x**5 + 28.346*x**4 - 30.982*x**3 + 19.802*x**2 - 7.3647*x + 1.8604,
        2.4: lambda x: 1.1155*x**6 - 6.7516*x**5 + 16.733*x**4 - 21.987*x**3 + 16.589*x**2 - 7.0242*x + 1.9508,
        2.8: lambda x: 1.1513*x**6 - 7.0938*x**5 + 17.952*x**4 - 24.034*x**3 + 18.257*x**2 - 7.6829*x + 2.0938,
        3.6: lambda x: 1.5272*x**6 - 8.4888*x**5 + 19.491*x**4 - 24.058*x**3 + 17.367*x**2 - 7.2931*x + 2.1322,
        4.4: lambda x: 1.1987*x**6 - 7.137*x**5 + 17.523*x**4 - 22.941*x**3 + 17.262*x**2 - 7.4016*x + 2.2127,
    },
    0.7: {
        0.4: lambda x: 4.7366*x**6 - 24.06*x**5 + 48.36*x**4 - 48.651*x**3 + 25.126*x**2 - 5.5821*x + 0.7802,
        0.8: lambda x: 4.0371*x**6 - 21.033*x**5 + 44.223*x**4 - 48.245*x**3 + 29*x**2 - 8.8891*x + 1.481,
        1.2: lambda x: 2.3264*x**6 - 12.602*x**5 + 27.808*x**4 - 32.413*x**3 + 21.533*x**2 - 7.8001*x + 1.6686,
        1.6: lambda x: 0.9411*x**6 - 5.6691*x**5 + 14.179*x**4 - 19.121*x**3 + 14.963*x**2 - 6.4941*x + 1.7272,
        2.0: lambda x: 2.0041*x**6 - 10.826*x**5 + 23.751*x**4 - 27.442*x**3 + 18.243*x**2 - 7.0385*x + 1.8536,
        2.4: lambda x: 0.151*x**6 - 1.4625*x**5 + 5.2042*x**4 - 9.3184*x**3 + 9.261*x**2 - 5.0225*x + 1.7696,
        2.8: lambda x: 0.8377*x**6 - 5.0027*x**5 + 12.394*x**4 - 16.551*x**3 + 12.962*x**2 - 5.9428*x + 1.9122,
        3.6: lambda x: 0.3223*x**6 - 2.5952*x**5 + 8.1263*x**4 - 13*x**3 + 11.579*x**2 - 5.7651*x + 1.9966,
        4.4: lambda x: 1.8415*x**6 - 10.293*x**5 + 23.435*x**4 - 28.157*x**3 + 19.424*x**2 - 7.7716*x + 2.2355,
    },
    0.6: {
        0.4: lambda x: 5.6861*x**6 - 29.165*x**5 + 59.597*x**4 - 61.632*x**3 + 33.492*x**2 - 8.4704*x + 1.0647,
        0.8: lambda x: 9.6468*x**6 - 48.711*x**5 + 97.658*x**4 - 99.242*x**3 + 54.152*x**2 - 15.144*x + 2.0837,
        1.2: lambda x: 6.9596*x**6 - 35.752*x**5 + 73.462*x**4 - 77.305*x**3 + 44.416*x**2 - 13.595*x + 2.2487,
        1.6: lambda x: 6.003*x**6 - 31.21*x**5 + 64.996*x**4 - 69.487*x**3 + 40.748*x**2 - 12.922*x + 2.3508,
        2.0: lambda x: 4.6777*x**6 - 24.405*x**5 + 51.016*x**4 - 54.817*x**3 + 32.55*x**2 - 10.751*x + 2.2472,
        2.4: lambda x: 4.2008*x**6 - 22.219*x**5 + 47.321*x**4 - 52.107*x**3 + 31.828*x**2 - 10.795*x + 2.3413,
        2.8: lambda x: 2.5599*x**6 - 13.915*x**5 + 30.716*x**4 - 35.535*x**3 + 23.283*x**2 - 8.7156*x + 2.2205,
        3.6: lambda x: 1.8325*x**6 - 10.122*x**5 + 22.824*x**4 - 27.189*x**3 + 18.602*x**2 - 7.4493*x + 2.1745,
        4.4: lambda x: 0.7434*x**6 - 4.7001*x**5 + 12.226*x**4 - 16.922*x**3 + 13.493*x**2 - 6.2532*x + 2.1318,
    },
    0.5: {
        0.4: lambda x: 5.1677*x**6 - 26.009*x**5 + 52.321*x**4 - 54.066*x**3 + 30.873*x**2 - 9.5937*x + 1.6907,
        0.8: lambda x: 5.1677*x**6 - 26.009*x**5 + 52.321*x**4 - 54.066*x**3 + 30.873*x**2 - 9.5937*x + 1.6907,
        1.2: lambda x: 2.1047*x**6 - 11.305*x**5 + 24.778*x**4 - 28.762*x**3 + 19.258*x**2 - 7.4437*x + 1.7879,
        1.6: lambda x: 1.9465*x**6 - 10.639*x**5 + 23.713*x**4 - 27.855*x**3 + 18.737*x**2 - 7.3442*x + 1.9134,
        2.0: lambda x: 2.0101*x**6 - 10.924*x**5 + 24.138*x**4 - 28.053*x**3 + 18.665*x**2 - 7.2832*x + 1.9858,
        2.4: lambda x: 0.8863*x**6 - 5.2966*x**5 + 13.134*x**4 - 17.473*x**3 + 13.507*x**2 - 6.1303*x + 1.9632,
        2.8: lambda x: 1.1188*x**6 - 6.4501*x**5 + 15.512*x**4 - 20.079*x**3 + 15.084*x**2 - 6.6282*x + 2.0769,
        3.6: lambda x: 1.1024*x**6 - 6.4614*x**5 + 15.689*x**4 - 20.372*x**3 + 15.283*x**2 - 6.7093*x + 2.1645,
        4.4: lambda x: -0.1239*x**6 - 0.1875*x**5 + 2.956*x**4 - 7.4124*x**3 + 8.4254*x**2 - 4.9697*x + 2.0549,
    },
    # Add the remaining B values and their corresponding T values and equations here...
}

# Define the equations for Amplitude Ratio (A) based on B and T values
equations_a = {
    1.0: {
        0.4: lambda x: 0.0461*x**6 + 0.8533*x**5 - 4.5885*x**4 + 8.0328*x**3 - 5.801*x**2 + 1.4785*x + 0.0054,
        0.8: lambda x: -0.3708*x**6 + 2.2462*x**5 - 5.2981*x**4 + 6.6634*x**3 - 4.9049*x**2 + 1.7131*x + 0.0095,
        1.2: lambda x: -0.5046*x**6 + 2.838*x**5 - 6.4337*x**4 + 7.7278*x**3 - 5.5243*x**2 + 2.1173*x - 0.0056,
        1.6: lambda x: -0.6221*x**6 + 2.8577*x**5 - 5.2225*x**4 + 5.257*x**3 - 3.7792*x**2 + 1.8523*x + 0.001,
        2.0: lambda x: -0.5562*x**6 + 2.7877*x**5 - 5.6166*x**4 + 6.0599*x**3 - 4.2822*x**2 + 2.0876*x - 0.006,
        2.4: lambda x: -0.2352*x**6 + 1.3234*x**5 - 3.0056*x**4 + 3.7478*x**3 - 3.2466*x**2 + 1.9922*x - 0.006,
        2.8: lambda x: -0.5159*x**6 + 2.5693*x**5 - 5.0964*x**4 + 5.2929*x**3 - 3.6415*x**2 + 2.0608*x - 0.0095,
        3.6: lambda x: -0.0352*x**6 + 0.333*x**5 - 1.0199*x**4 + 1.5812*x**3 - 1.8448*x**2 + 1.7579*x + 0.0059,
        4.4: lambda x: 0.3287*x**6 - 1.2652*x**5 + 1.57*x**4 - 0.3027*x**3 - 1.225*x**2 + 1.7603*x - 0.0003,
    },
    0.9: {
        0.4: lambda x: -1.311*x**6 + 6.3001*x**5 - 12.157*x**4 + 12.221*x**3 - 6.6985*x**2 + 1.6551*x + 0.0107,
        0.8: lambda x: -0.5593*x**6 + 2.8897*x**5 - 6.015*x**4 + 6.7368*x**3 - 4.6309*x**2 + 1.7593*x + 0.0198,
        1.2: lambda x: -0.4747*x**6 + 2.5555*x**5 - 5.589*x**4 + 6.4657*x**3 - 4.5458*x**2 + 1.9698*x + 0.0112,
        1.6: lambda x: -1.1465*x**6 + 5.4959*x**5 - 10.152*x**4 + 9.331*x**3 - 5.0894*x**2 + 2.0875*x + 0.0047,
        2.0: lambda x: -0.6847*x**6 + 3.3489*x**5 - 6.368*x**4 + 6.1438*x**3 - 3.7704*x**2 + 1.9732*x + 0.0046,
        2.4: lambda x: -0.003*x**6 + 0.1739*x**5 - 0.7371*x**4 + 1.3841*x**3 - 1.8133*x**2 + 1.7157*x + 0.0169,
        2.8: lambda x: -0.1473*x**6 + 0.9731*x**5 - 2.4792*x**4 + 3.2207*x**3 - 2.7131*x**2 + 1.9431*x + 0.0043,
        3.6: lambda x: -0.7344*x**6 + 3.7509*x**5 - 7.3772*x**4 + 7.1115*x**3 - 3.9617*x**2 + 2.1039*x + 0.0006,
        4.4: lambda x: -0.4357*x**6 + 2.2455*x**5 - 4.437*x**4 + 4.351*x**3 - 2.6953*x**2 + 1.913*x + 0.0119,
    },
    0.8: {
        0.4: lambda x: -0.5626*x**6 + 3.2143*x**5 - 7.4153*x**4 + 8.7777*x**3 - 5.5886*x**2 + 1.6564*x + 0.008,
        0.8: lambda x: -0.5731*x**6 + 2.909*x**5 - 6.0925*x**4 + 6.9115*x**3 - 4.7894*x**2 + 1.9634*x + 0.0086,
        1.2: lambda x: -0.2289*x**6 + 1.4366*x**5 - 3.5996*x**4 + 4.7175*x**3 - 3.806*x**2 + 2.0199*x + 0.0063,
        1.6: lambda x: -0.5701*x**6 + 2.8884*x**5 - 5.8551*x**4 + 6.1691*x**3 - 4.0447*x**2 + 2.0934*x + 0.0069,
        2.0: lambda x: -0.0807*x**6 + 0.6157*x**5 - 1.7721*x**4 + 2.6346*x**3 - 2.5748*x**2 + 1.9592*x + 0.0066,
        2.4: lambda x: 0.1998*x**6 - 0.711*x**5 + 0.6255*x**4 + 0.5535*x**3 - 1.6548*x**2 + 1.8384*x + 0.008,
        2.8: lambda x: -0.0042*x**6 + 0.1142*x**5 - 0.5064*x**4 + 1.0646*x**3 - 1.5505*x**2 + 1.7851*x + 0.0122,
        3.6: lambda x: -1.0058*x**6 + 4.4553*x**5 - 7.5003*x**4 + 6.1561*x**3 - 3.0977*x**2 + 1.9571*x + 0.0085,
        4.4: lambda x: -1.0023*x**6 + 4.6889*x**5 - 8.3919*x**4 + 7.258*x**3 - 3.551*x**2 + 2.0105*x + 0.0089,
    },
    0.7: {
        0.4: lambda x: -0.145*x**6 + 1.0464*x**5 - 3.0694*x**4 + 4.6561*x**3 - 3.8342*x**2 + 1.4754*x + 0.0174,
        0.8: lambda x: -0.763*x**6 + 3.704*x**5 - 7.188*x**4 + 7.3528*x**3 - 4.6533*x**2 + 1.9904*x + 0.0022,
        1.2: lambda x: -0.4798*x**6 + 2.403*x**5 - 4.787*x**4 + 5.0033*x**3 - 3.3994*x**2 + 1.918*x + 0.0026,
        1.6: lambda x: -0.7422*x**6 + 3.6281*x**5 - 6.9552*x**4 + 6.7841*x**3 - 4.0285*x**2 + 2.1035*x + 0.0011,
        2.0: lambda x: -0.5131*x**6 + 2.5462*x**5 - 4.94*x**4 + 4.9258*x**3 - 3.1675*x**2 + 2.0307*x - 0.0006,
        2.4: lambda x: -0.2055*x**6 + 1.0019*x**5 - 1.9638*x**4 + 2.1876*x**3 - 1.9341*x**2 + 1.8467*x + 0.0015,
        2.8: lambda x: -0.2981*x**6 + 1.2984*x**5 - 2.2039*x**4 + 2.0852*x**3 - 1.7116*x**2 + 1.8004*x + 0.004,
        3.6: lambda x: -0.5051*x**6 + 2.414*x**5 - 4.5186*x**4 + 4.3276*x**3 - 2.6356*x**2 + 1.9376*x + 0.0053,
        4.4: lambda x: -0.8015*x**6 + 3.6177*x**5 - 6.3199*x**4 + 5.5439*x**3 - 2.9928*x**2 + 2.0097*x + 0.0021,
    },
    0.6: {
        0.4: lambda x: -0.2506*x**6 + 1.284*x**5 - 2.8237*x**4 + 3.6462*x**3 - 3.0463*x**2 + 1.3782*x + 0.0245,
        0.8: lambda x: -0.1876*x**6 + 1.0382*x**5 - 2.3133*x**4 + 2.9136*x**3 - 2.615*x**2 + 1.7143*x + 0.0073,
        1.2: lambda x: -0.0234*x**6 + 0.2732*x**5 - 0.8651*x**4 + 1.358*x**3 - 1.5764*x**2 + 1.5908*x + 0.0161,
        1.6: lambda x: -0.0306*x**6 - 0.0312*x**5 + 0.3739*x**4 - 0.3611*x**3 - 0.5474*x**2 + 1.4541*x + 0.0196,
        2.0: lambda x: 0.4711*x**6 - 2.384*x**5 + 4.6172*x**4 - 4.0462*x**3 + 1.0397*x**2 + 1.2275*x + 0.0261,
        2.4: lambda x: -0.1005*x**6 + 0.0873*x**5 + 0.6671*x**4 - 1.2457*x**3 + 0.2482*x**2 + 1.3213*x + 0.023,
        2.8: lambda x: 0.2103*x**6 - 1.1287*x**5 + 2.3089*x**4 - 2.0943*x**3 + 0.3866*x**2 + 1.3214*x + 0.0239,
        3.6: lambda x: -0.2792*x**6 + 1.1965*x**5 - 1.9116*x**4 + 1.5307*x**3 - 1.0569*x**2 + 1.5717*x + 0.0102,
        4.4: lambda x: 0.3095*x**6 - 1.3349*x**5 + 2.1599*x**4 - 1.5238*x**3 + 0.0786*x**2 + 1.3706*x + 0.0236,
    },
    0.5: {
        0.4: lambda x: -0.5909*x**6 + 2.9365*x**5 - 6.008*x**4 + 6.7359*x**3 - 4.7231*x**2 + 1.9564*x + 0.0164,
        0.8: lambda x: -0.9686*x**6 + 4.612*x**5 - 8.6512*x**4 + 8.307*x**3 - 4.7907*x**2 + 2.1693*x + 0.0095,
        1.2: lambda x: -1.135*x**6 + 5.3262*x**5 - 9.8018*x**4 + 9.1199*x**3 - 4.9369*x**2 + 2.3014*x - 0.0009,
        1.6: lambda x: 0.1457*x**6 - 0.5621*x**5 + 0.4986*x**4 + 0.6738*x**3 - 1.71*x**2 + 1.9076*x + 0.0137,
        2.0: lambda x: -0.1786*x**6 + 0.8821*x**5 - 1.9113*x**4 + 2.4802*x**3 - 2.2361*x**2 + 1.9766*x + 0.0082,
        2.4: lambda x: 0.0776*x**6 - 0.3608*x**5 + 0.4274*x**4 + 0.3481*x**3 - 1.2887*x**2 + 1.8022*x + 0.0159,
        2.8: lambda x: -0.1059*x**6 + 0.5761*x**5 - 1.3956*x**4 + 1.996*x**3 - 1.9214*x**2 + 1.8951*x + 0.0126,
        3.6: lambda x: -0.8929*x**6 + 4.2016*x**5 - 7.7841*x**4 + 7.3182*x**3 - 3.9598*x**2 + 2.2181*x - 0.0002,
        4.4: lambda x: -0.9046*x**6 + 4.1589*x**5 - 7.5223*x**4 + 6.8901*x**3 - 3.6452*x**2 + 2.1376*x + 0.0037,
    },
    # Add the remaining B values and their corresponding T values and equations here...
}

# Function to interpolate and calculate C
def calculate_c(b, t, x):
    # Collect all B and T values for interpolation
    b_values = sorted(equations_c.keys())
    t_values = sorted(equations_c[b_values[0]].keys())

    # Initialize an empty array to hold the result of C for the interpolation grid
    c_values = np.zeros((len(b_values), len(t_values)))

    # Populate the C values using the equations
    for i, b_val in enumerate(b_values):
        for j, t_val in enumerate(t_values):
            if t_val in equations_c[b_val]:
                try:
                    c_values[i, j] = equations_c[b_val][t_val](x)
                except Exception as e:
                    print(f"Error computing C for B={b_val}, T={t_val}, x={x}: {e}")
                    c_values[i, j] = np.nan
            else:
                # Handle missing T values gracefully, here we simply use NaN
                c_values[i, j] = np.nan

    # Interpolate the C value for the given B and T
    interp_func = RegularGridInterpolator((b_values, t_values), c_values, method='linear', bounds_error=False, fill_value=np.nan)
    return interp_func((b, t))

# Function to interpolate and calculate A
def calculate_a(b, t, x):
    # Collect all B and T values for interpolation
    b_values = sorted(equations_a.keys())
    t_values = sorted(equations_a[b_values[0]].keys())

    # Initialize an empty array to hold the result of A for the interpolation grid
    a_values = np.zeros((len(b_values), len(t_values)))

    # Populate the A values using the equations
    for i, b_val in enumerate(b_values):
        for j, t_val in enumerate(t_values):
            if t_val in equations_a[b_val]:
                try:
                    a_values[i, j] = equations_a[b_val][t_val](x)
                except Exception as e:
                    print(f"Error computing A for B={b_val}, T={t_val}, x={x}: {e}")
                    a_values[i, j] = np.nan
            else:
                # Handle missing T values gracefully, here we simply use NaN
                a_values[i, j] = np.nan

    # Interpolate the A value for the given B and T
    interp_func = RegularGridInterpolator((b_values, t_values), a_values, method='linear', bounds_error=False, fill_value=np.nan)
    return interp_func((b, t))

# Continuous input and calculation loop
def main():
    print("Continuous C and A Value Calculator")
    print("Press Ctrl+C or type 'exit' at any prompt to quit.\n")
    while True:
        try:
            # Get user inputs for B, T, and X with error handling
            b_input = input("Enter the value for Bn (or type 'exit' to quit): ").strip()
            if b_input.lower() == 'exit':
                print("Exiting the calculator. Goodbye!")
                break
            b = float(b_input)

            t_input = input("Enter the value for B/T (or type 'exit' to quit): ").strip()
            if t_input.lower() == 'exit':
                print("Exiting the calculator. Goodbye!")
                break
            t = float(t_input)

            x_input = input("Enter the value for x (or type 'exit' to quit): ").strip()
            if x_input.lower() == 'exit':
                print("Exiting the calculator. Goodbye!")
                break
            x = float(x_input)

            # Calculate C and A
            c = calculate_c(b, t, x)
            a = calculate_a(b, t, x)

            # Prepare output messages
            c_message = f"For Bn: {b}, B/T: {t}, x: {x} --> Added Mass Coefficient (C): {c}" if not np.isnan(c) else f"For Bn: {b}, B/T: {t}, x: {x} --> Added Mass Coefficient (C): Not available (out of interpolation bounds)"
            a_message = f"For Bn: {b}, B/T: {t}, x: {x} --> Amplitude Ratio (A): {a}" if not np.isnan(a) else f"For Bn: {b}, B/T: {t}, x: {x} --> Amplitude Ratio (A): Not available (out of interpolation bounds)"

            # Print the results
            print(c_message)
            print(a_message + "\n")
        except ValueError:
            print("Invalid input. Please enter numerical values only.\n")
        except KeyboardInterrupt:
            print("\nExiting the calculator. Goodbye!")
            break
        except Exception as e:
            print(f"An unexpected error occurred: {e}\n")

if __name__ == "__main__":
    main()

# -*- coding: utf-8 -*-
"""
Created on Sat Nov  7 15:47:16 2020

@author: erika
"""


import numpy as np
import matplotlib.pyplot as plt
from template_figure import axes_fig
import matplotlib.animation as animation
from celluloid import Camera
  



def periodic(N,f):
# #FONCTION IMPOSANT PÉRIODICITÉ HORIZONTALE ET VERTICALE
    f[1:N+1,0] = f[1:N+1,N-1] # Periodicite horizontale
    f[1:N+1,N+1] = f[1:N+1,2]
    f[0,1:N+1] = f[N-1,1:N+1] # Periodicite verticale
    f[N+1,1:N+1] = f[2,1:N+1]


#Fonction qui calcule le terme source  
def source(x,y,x0,y0,S0):
    w  = 0.25
    r2 = (x-x0)**2 + (y-y0)**2
    S = S0*np.exp(-r2/w**2)

    return S
#----------- VITESSE EN X ET Y --------------#


def vitesse_x(y,t):
    #Écoulement en x (2.83) 
    A = np.sqrt(6)
    epsilon,omega = 1,5

    return A*np.cos( y + epsilon*np.sin(omega*t))


def vitesse_y(x,t):
    #Écoulement en y (2.84) 
    B = np.sqrt(6)
    epsilon,omega = 1,5
        
    return B*np.sin( x + epsilon*np.cos(omega*t))
#---------------------------------------------#


def advectif_LaxWendroff(t,x,y,c,delta_t,delta_x,N):
    """
    Fonction qui calcule le terme advectif avec la méthode de Lax-Wendroff
    """

    delta_y = delta_x
    #Écoulement en x et en y (2.83) et (2.84)
    uxkp = vitesse_x(y+delta_y/2, t+delta_t/2)           # k+1/2
    uxkm = vitesse_x(y-delta_y/2,t+delta_t/2)            # k-1/2
    uyjp = vitesse_y(x+delta_x/2,t+delta_t/2)            # j+1/2
    uyjm = vitesse_y(x-delta_x/2,t+delta_t/2)            # j-1/2
   
    c_demijpkp = 1/4*(c[2:N+2,1:N+1] + c[1:N+1,1:N+1] + c[2:N+2,2:N+2] + c[1:N+1,2:N+2]) - delta_t/2*( uxkp*(c[2:N+2,1:N+1] - c[1:N+1,1:N+1])/delta_x + uyjp*(c[1:N+1,2:N+2] - c[1:N+1,1:N+1])/delta_y)  #j+1, k+1 Éqn (2.78)
    c_demijmkp = 1/4*(c[0:N,1:N+1] + c[1:N+1,1:N+1] + c[0:N,2:N+2] + c[1:N+1,2:N+2]) - delta_t/2*( -uxkp*(c[0:N,1:N+1] - c[1:N+1,1:N+1])/delta_x + uyjm*(c[1:N+1,2:N+2] - c[1:N+1,1:N+1])/delta_y)  #j-1, k+1
    c_demijpkm = 1/4*(c[2:N+2,1:N+1] + c[1:N+1,1:N+1] + c[2:N+2,0:N] + c[1:N+1,0:N]) - delta_t/2*( uxkm*(c[2:N+2,1:N+1] - c[1:N+1,1:N+1])/delta_x - uyjp*(c[1:N+1,0:N] - c[1:N+1,1:N+1])/delta_y)  #j+1, k-1 
    c_demijmkm = 1/4*(c[0:N,1:N+1] + c[1:N+1,1:N+1] + c[0:N,0:N] + c[1:N+1,0:N]) - delta_t/2*( -uxkm*(c[0:N,1:N+1] - c[1:N+1,1:N+1])/delta_x - uyjm*(c[1:N+1,0:N] - c[1:N+1,1:N+1])/delta_y)  #j-1, k-1

    Fxjpkp = uxkp*c_demijpkp # j+1/2, k+1/2
    Fxjmkp = uxkp*c_demijmkp # j-1/2, k+1/2
    Fxjpkm = uxkm*c_demijpkm # j+1/2, k-1/2
    Fxjmkm = uxkm*c_demijmkm # j-1/2, k-1/2

    Fyjpkp = uyjp*c_demijpkp # j+1/2, k+1/2
    Fyjmkp = uyjm*c_demijmkp # j-1/2, k+1/2
    Fyjpkm = uyjp*c_demijpkm # j+1/2, k-1/2
    Fyjmkm = uyjm*c_demijmkm # j-1/2, k-1/2     
    
    F = -1/2*delta_t/delta_x*(Fxjpkm - Fxjmkm + Fxjpkp - Fxjmkp) - 1/2*delta_t/delta_y*(Fyjpkp - Fyjpkm + Fyjmkp - Fyjmkm)
    return F    
#FIN TERME ADVECTIF AVEC LAX-WENDROFF


def boucle_temporelle_LaxWendroff(x0 = 0.5*2*np.pi, y0 = 0.5*2*np.pi,S0=2.,td=0.01,NITER=3001,tf=1.75):
    N = 401                            # Taille de la maille physique (4 fois plus grand pour Lax-Wendroff)
    delta_x = 2*np.pi/(N-1)            #pas en x,  = delta_y
    
    c       = np.zeros((N+2,N+2))      # maille physique N par N avec noeuds fantômes
    cnp1    = np.zeros((N+2,N+2))      # même chose au pas de temps n+1
    gros_vecteur= np.zeros((N+2,N+2,NITER))
    Pe = 10**3                         # nombre de Peclet

    delta_t  = 0.001

     
    t        = np.arange(0,NITER*delta_t,delta_t)              #array du temps
    xx        = np.arange(0,(N)*delta_x,delta_x)               #array de la position en x
    yy        = np.arange(0,(N)*delta_x,delta_x)               #array de la position en y
    
    #Vectorise x et y
    x = np.transpose(np.ones((N,N))*xx)
    y = np.ones((N,N))*yy
    
    #Calcul du terme source avant d'entrer dans la boucle temporelle
    Source = source(x,y,x0,y0,S0)
    avant_polluant,polluant, apres_polluant = True, True, True


    for iter in range(1,NITER-1): #boucle itérant sur le temps
        
        if (t[iter]+delta_t/2 > tf and avant_polluant == True):
            S = Source*0
            avant_polluant = False
        
        elif(t[iter]+delta_t/2 < td and apres_polluant == True):
            S = Source*0
            apres_polluant = False
        
        elif(t[iter]+delta_t/2 >= td and polluant == True):
            S = Source
            polluant = False
        # if iter%50 == 0:
        #     print(iter)
            
        cnp1[1:N+1,1:N+1] = c[1:N+1,1:N+1] + advectif_LaxWendroff(t[iter],x,y,c,delta_t,delta_x,N) + delta_t/(Pe*(delta_x)**2)*(c[1:N+1,2:N+2] + c[2:N+2,1:N+1] - 4*c[1:N+1,1:N+1] + c[1:N+1,0:N] + c[0:N,1:N+1]) + delta_t*S

        periodic(N,cnp1)     # impose la périodicité
        c  = np.copy(cnp1)   # mise à jour de c pour le prochain pas de temps
        gros_vecteur[:,:,iter] = np.copy(cnp1)
        
    return t[:], gros_vecteur

  #  return t[1:], c_x25y75, c_x50y50, c_x75y25




N = 401                       # Taille de la maille physique
delta_x = 2*np.pi/(N-1)       #pas en x,  = delta_y
x0 = 3.97722312
y0 = 0.46915039
S0 = 4.8989023
td = 0.05521642

A = np.loadtxt('Donnees/donnees_projet2.txt',unpack=True)
tk        = A[0]
ck_x75y25 = A[1]
ck_x50y50 = A[2]
ck_x25y75 = A[3]

N=401
c_init = np.zeros((N+2,N+2))      # maille physique N par N avec noeuds fantômes
c_init[75*4+1,25*4+1] = ck_x75y25[500]

t_S,gros_vecteur = boucle_temporelle_LaxWendroff()




def ANIM(t_S,gros_vecteur,nom_fichier):
    fig = plt.figure(0)
    camera = Camera(fig)
    for i in range(round(len(t_S)/10)):
        with plt.xkcd():
            titles = 'time = '+str(np.round(t_S[i*10+9],2))+' s'
            plt.imshow(np.flip(np.transpose(gros_vecteur[:,:,i*10+9]),0),extent=[0,1,0,1],cmap='coolwarm')
         #   plt.plot(0.71,.08,'rx')
          #  plt.plot(0.25,0.75,'o',color='purple')
          #  plt.plot(0.5,0.5,'o',color='blue')
            plt.plot(0.5,0.5,'go')
            plt.text(0.3,1.05,titles)
            camera.snap()
        print(i)
        
    animation = camera.animate(interval = 150)
    animation.save('freefall.gif')


#ANIM(t_S,gros_vecteur[:,:,:],'freefal.gif')

ANIM(t_S,gros_vecteur[:,:,:],'sans_vitesse.gif')


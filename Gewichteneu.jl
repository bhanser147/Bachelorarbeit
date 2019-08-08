using DataFrames
using Distributions
using Random, Flux
using Plots

function design(n) #n Patienten
    #Variablen initialisieren
    x_G1 = rand(n)
    x_G2 = rand(n)
    xT = rand(n)
    x = rand(n, 10) #x_1 bis x_10
    yC = zeros(n)
    yE = zeros(n)
    y = rand(n)

    for i = 1:n
        x_G1[i] = Int.(bitrand(1))[1] #P(x_G1 = 1) = 1/2
        x_G2[i] = Int.(bitrand(1))[1] #P(x_G2 = 1) = 1/2
    end
    #println("x_G1", x_G1)
    #println("x_G2", x_G2)

    for i = 1:n
        for j = 1:5 #x_1 - x_5 bekommen ihre Wahrscheinlichkeiten zugewiesen
            if x_G1[i] == 0 #Wenn x_G1 = 0, hat die Person eher keine Störfaktoren
                 x[i,j] = rand(Bernoulli(0.2),1)[1] #P(x_i)
                 xT[i] = rand(Bernoulli(0.8),1)[1]
             else
                 x[i,j] = rand(Bernoulli(0.8),1)[1]
                 xT[i] = rand(Bernoulli(0.2),1)[1]
                 yC[i] = 0.5
             end
         end
         for j = 6:10 #x_6-x_10 bekommen ihre Wahrscheinlichkeiten zugewiesen
            if x_G2[i] == 0
                 x[i,j] = rand(Bernoulli(0.2),1)[1]
             else
                 x[i,j] = rand(Bernoulli(0.8),1)[1]
             end
        end
        if (sum(x[i,j] for j = 6:10)>2)
            yE[i] = xT[i]*0.5 #?
        end
        if xT[i] == 0
            y[i]=rand(Normal(0,1))
        else
            y[i]=rand(Normal(1,1))
        end
        y[i] = y[i] + yE[i] + yC[i]
    end
    yp = deepcopy(minimum(y))
    if yp < 0
        y = y .+ abs(yp)
    end
    return x, y, x_G1, x_G2, xT
end

binsum(v) = Int(v[1] + v[2]*2 + v[3]*4 + v[4]*8 + v[5]*16)

function GewichtNN(n)
#Ergebnis ist P(a = 1, L = l) v
    x, y, x_G1, x_G2, xT = design(n)

    data = Vector(undef, n)

    weights = Chain(Dense(5, 5, relu), Dense(5,1,relu))  #berechnet bedingte Wahrscheinlichkeit
    loss(x, y) = Flux.mse(weights(x), y)

    #Anzahl pro Kombination von Störfaktoren von xT = 1 und xT = 0
    counter_treatment = zeros(32)
    counter_notreatment = zeros(32)

    x1_5 = x[:,1:5]

    for i in 1:n #
        if xT[i] == 1
            counter_treatment[binsum(x1_5[i,:]) + 1] += 1
        else
            counter_notreatment[binsum(x1_5[i,:]) + 1] += 1
        end
    end

    pxT0L = Vector(undef, 32)

    for i in 1:32
        if (counter_notreatment[i] + counter_treatment[i]) > 0
            pxT0L[i] = counter_notreatment[i]/(counter_notreatment[i] + counter_treatment[i])
        else
            pxT0L[i] = 0
        end
    end


    for i = 1:n
        if xT[i] == 1
            data[i] = (x[i,1:5], 1 - pxT0L[binsum(x1_5[i,:]) + 1])
        else
            data[i] = (x[i,1:5], pxT0L[binsum(x1_5[i,:]) + 1])
    end

    params = Flux.params(weights)
    opt = Descent(0.01)

    Flux.@epochs 20 Flux.train!(loss, params, data, opt)

    return weights
end

function Gewichte(xT, x, weights, n)

    W = Vector(undef, n)

    pxT = sum(xT)
    pxT0 = (n-pxT)/n
    pxT1 = pxT/n

    for i in 1:n
        if xT[i] == 1
            W[i] = pxT1/(Tracker.data(weights(x[i, 1:5])))[1]
        else
            W[i] = pxT0/(Tracker.data(weights(x[i,1:5])))[1]
        end
    end

    for i in 1:n
        if W[i] > 10
            W[i] = 10
        end
    end

return W
end



h(x) = min(0,(1 .- x))
h1(x) = exp.(-x)
j(x,z) = (2*x.-1).*(sign.(z .- 0.5))
loss2(x, y, z) = sum((h1(j(x,z)).*y) ./ ((x*(sum(x)/length(x))) + ((ones(length(x)) .- x)*(1 - sum(x)/length(x))))) / length(x)

loss1(x, y, z) = sum((abs.(x .- round.(z)).*y) ./ ((x*(sum(x)/length(x))) + ((ones(length(x)) .- x)*(1 - sum(x)/length(x))))) / length(x)

function nn(n)
 #  n = 10#n Anzahl an Patienten

x, y, x_G1, x_G2, xT = design(n)

#weights = GewichtNN(n, x, xT)
weights = GewichtNN(n)
W = Vector(undef, n)
for i in 1:n
    W[i] = Tracker.data(weights(x[i,1:5]))[1]
end

println("Gewichte fertig")

data = Vector(undef, n)
dataw = Vector(undef, n)
datawa = Vector(undef, n)


modelw = Chain(Dense(10,10,relu), Dense(10,1,relu))
#modelwa = Chain(Dense(10,10,relu), Dense(10,1,relu))
model = Chain(Dense(10,10,relu), Dense(10,1,relu))


for i in 1:n
    data[i] = ([xT[i]],[y[i]], x[i,:])
    dataw[i] = ([xT[i]],[y[i]*W[i]], x[i,:])
    #datawa[i] = ([xT[i]],[y[i]*Wa[i]], x[i,:])
end
#println(dataw)

    #data = (xT, y, x)
L(x, y, z) = loss2(x, y, model(z))
opt = Descent(0.01)
params = Flux.params(model)
paramsw = Flux.params(modelw)
#paramswa = Flux.params(modelwa)
Flux.@epochs 50 Flux.train!(L, params, data, opt)
println("Erstes NN fertig")
Flux.@epochs 50 Flux.train!(L, paramsw, dataw, opt)
println("Zweites NN fertig")
#Flux.@epochs 50 Flux.train!(L, paramswa, datawa, opt)
#println("Zweites NN fertig")
    return model, modelw
end

function m(x, y, x_G1, x_G2, xT, model, model1)
    n = length(x_G1)
    sx_G1 = (sum(x_G1))
    sx_G2 = (sum(x_G2))
    a, b, c, d, e, f, g, h = 0, 0, 0, 0, 0, 0, 0, 0

    d_G1_xT_0, d_G1_xT_1, d_G2_xT_0, d_G2_xT_1 = 0, 0, 0, 0
    m_x_G1_0_xT_0 = Vector(undef, 0)
    m_x_G1_0_xT_1 = Vector(undef, 0)
    m_x_G1_1_xT_0 = Vector(undef, 0)
    m_x_G1_1_xT_1 = Vector(undef, 0)
    mw_x_G1_0_xT_0 = Vector(undef, 0)
    mw_x_G1_0_xT_1 = Vector(undef, 0)
    mw_x_G1_1_xT_0 = Vector(undef, 0)
    mw_x_G1_1_xT_1 = Vector(undef, 0)
    m_x_G2_0_xT_0 = Vector(undef, 0)
    m_x_G2_0_xT_1 = Vector(undef, 0)
    m_x_G2_1_xT_0 = Vector(undef, 0)
    m_x_G2_1_xT_1 = Vector(undef, 0)
    mw_x_G2_0_xT_0 = Vector(undef, 0)
    mw_x_G2_0_xT_1 = Vector(undef, 0)
    mw_x_G2_1_xT_0 = Vector(undef, 0)
    mw_x_G2_1_xT_1 = Vector(undef, 0)

    m_G1_0_w_xT_0, m_G1_0_w_xT_1, m_G1_0_wa_xT_0, m_G1_0_wa_xT_1 = 0, 0, 0, 0
    m_G1_1_w_xT_0, m_G1_1_w_xT_1, m_G1_1_wa_xT_0, m_G1_1_wa_xT_1 = 0, 0, 0, 0
    m_G1_0_xT_0, m_G1_0_xT_1 = 0, 0
    m_G1_1_xT_0, m_G1_1_xT_1 = 0, 0
    m_G2_0_w_xT_0, m_G2_0_w_xT_1, m_G2_0_wa_xT_0, m_G2_0_wa_xT_1 = 0, 0, 0, 0
    m_G2_1_w_xT_0, m_G2_1_w_xT_1, m_G2_1_wa_xT_0, m_G2_1_wa_xT_1 = 0, 0, 0, 0
    m_G2_0_xT_0, m_G2_0_xT_1 = 0, 0
    m_G2_1_xT_0, m_G2_1_xT_1  = 0, 0


    for i in 1:n
        if xT[i]==0
            if x_G1[i] == 0
                m_G1_0_xT_0 += Tracker.data(model(x[i,:]))[1]
                m_G1_0_w_xT_0 += Tracker.data(model1(x[i,:]))[1]
                #m_G1_0_wa_xT_0 += Tracker.data(model2(x[i,:]))[1]
                #append!(m_x_G1_0_xT_0, Tracker.data(model(x[i,:]))[1])
                #append!(mw_x_G1_0_xT_0, Tracker.data(model1(x[i,:]))[1])
                a += 1
            else
                m_G1_1_xT_0 += Tracker.data(model(x[i,:]))[1]
                m_G1_1_w_xT_0 += Tracker.data(model1(x[i,:]))[1]
                #m_G1_1_wa_xT_0 += Tracker.data(model2(x[i,:]))[1]
                #append!(m_x_G1_1_xT_0, Tracker.data(model(x[i,:]))[1])
                #append!(mw_x_G1_1_xT_0, Tracker.data(model1(x[i,:]))[1])
                b +=1
            end
            if x_G2[i] == 0
                m_G2_0_xT_0 += Tracker.data(model(x[i,:]))[1]
                m_G2_0_w_xT_0 += Tracker.data(model1(x[i,:]))[1]
                #m_G2_0_wa_xT_0 += Tracker.data(model2(x[i,:]))[1]
                #append!(m_x_G2_0_xT_0, Tracker.data(model(x[i,:]))[1])
                #append!(mw_x_G2_0_xT_0, Tracker.data(model1(x[i,:]))[1])
                c += 1
            else x_G2[i] == 1
                m_G2_1_xT_0 += Tracker.data(model(x[i,:]))[1]
                m_G2_1_w_xT_0 += Tracker.data(model1(x[i,:]))[1]
                #m_G2_1_wa_xT_0 += Tracker.data(model2(x[i,:]))[1]
                #append!(m_x_G2_1_xT_0, Tracker.data(model(x[i,:]))[1])
                #append!(mw_x_G2_1_xT_0, Tracker.data(model1(x[i,:]))[1])
                d += 1
            end
        else
            if x_G1[i] == 0
                m_G1_0_xT_1 += Tracker.data(model(x[i,:]))[1]
                m_G1_0_w_xT_1 += Tracker.data(model1(x[i,:]))[1]
                #m_G1_0_wa_xT_1 += Tracker.data(model2(x[i,:]))[1]
                #append!(m_x_G1_0_xT_1, Tracker.data(model(x[i,:]))[1])
                #append!(mw_x_G1_0_xT_1, Tracker.data(model1(x[i,:]))[1])
                e += 1
            else
                m_G1_1_xT_1 += Tracker.data(model(x[i,:]))[1]
                m_G1_1_w_xT_1 += Tracker.data(model1(x[i,:]))[1]
                #m_G1_1_wa_xT_1 += Tracker.data(model2(x[i,:]))[1]
                #append!(m_x_G1_1_xT_1, Tracker.data(model(x[i,:]))[1])
                #append!(mw_x_G1_1_xT_1, Tracker.data(model1(x[i,:]))[1])
                f += 1
            end
            if x_G2[i] == 0
                m_G2_0_xT_1 += Tracker.data(model(x[i,:]))[1]
                m_G2_0_w_xT_1 += Tracker.data(model1(x[i,:]))[1]
                #m_G2_0_wa_xT_1 += Tracker.data(model2(x[i,:]))[1]
                #append!(m_x_G2_0_xT_1, Tracker.data(model(x[i,:]))[1])
                #append!(mw_x_G2_0_xT_1, Tracker.data(model1(x[i,:]))[1])
                g += 1
            else x_G2[i] == 1
                m_G2_1_xT_1 += Tracker.data(model(x[i,:]))[1]
                m_G2_1_w_xT_1 += Tracker.data(model1(x[i,:]))[1]
                #m_G2_0_wa_xT_1 += Tracker.data(model2(x[i,:]))[1]
                #append!(m_x_G2_1_xT_1, Tracker.data(model(x[i,:]))[1])
                #append!(mw_x_G2_1_xT_1, Tracker.data(model1(x[i,:]))[1])
                h += 1
            end
        end
    end
     d_G1_xT_0 = abs(m_G1_0_xT_0/a-m_G1_1_xT_0/b)-abs(m_G1_0_w_xT_0/a- m_G1_1_w_xT_0/b)
     d_G1_xT_1 = abs(m_G1_0_xT_1/e-m_G1_1_xT_1/f)-abs(m_G1_0_w_xT_1/e- m_G1_1_w_xT_1/f)
     d_G2_xT_0 = abs(m_G2_0_xT_0/c-m_G2_1_xT_0/d)-abs(m_G2_0_w_xT_0/c- m_G2_1_w_xT_0/d)
     d_G2_xT_1 = abs(m_G2_0_xT_1/g-m_G2_1_xT_1/h)-abs(m_G2_0_w_xT_1/g- m_G2_1_w_xT_1/h)
      #zuerst Mittelwert, brauche Länge der G1 = 0 wenn xT = 0!!


    m= [abs(m_G1_0_xT_0/a-m_G1_1_xT_0/b) abs(m_G1_0_w_xT_0/a-m_G1_1_w_xT_0/b) d_G1_xT_0 abs(m_G2_0_xT_0/c-m_G2_1_xT_0/d) abs(m_G2_0_w_xT_0/c-m_G2_1_w_xT_0/d) d_G2_xT_0; abs(m_G1_0_xT_1/e-m_G1_1_xT_1/f) abs( m_G1_0_w_xT_1/e-m_G1_1_w_xT_1/f) d_G1_xT_1 abs(m_G2_0_xT_1/g-m_G2_1_xT_1/h) abs(m_G2_0_w_xT_1/g-m_G2_1_w_xT_1/h) d_G2_xT_1]

    #m = [abs(m_G1_0_xT_0/a-m_G1_1_xT_0/b) abs(m_G1_0_w_xT_0/a-m_G1_1_w_xT_0/b) abs(m_G1_0_wa_xT_0/a-m_G1_1_wa_xT_0/b) abs(m_G2_0_xT_0/c-m_G2_1_xT_0/d) abs(m_G2_0_w_xT_0/c-m_G2_1_w_xT_0/d) abs(m_G2_0_wa_xT_0/c-m_G2_1_wa_xT_0/d); abs(m_G1_0_xT_1/e-m_G1_1_xT_1/f) abs(m_G1_0_w_xT_1/e-m_G1_1_w_xT_1/f) abs(m_G1_0_wa_xT_1/e-m_G1_1_wa_xT_1/f) abs(m_G2_0_xT_1/g-m_G2_1_xT_1/h) abs(m_G2_0_w_xT_1/g-m_G2_1_w_xT_1/h) abs(m_G2_0_wa_xT_1/g-m_G2_1_wa_xT_1/h)]

    #s_G1_0_w = sum((mw_x_G1_0 .- m_G1_0).^2)/a
    #s_G1_1_w = sum((mw_x_G1_1 .- m_G1_0).^2)/sx_G1
    #s_G1_0 = sum((m_x_G1_0 .- m_G1_0).^2)/a
    #s_G1_1 = sum((m_x_G1_1 .- m_G1_0).^2)/sx_G1
    #s_G2_0_w = sum((mw_x_G2_0 .- m_G1_0).^2)/b
    #s_G2_1_w = sum((mw_x_G2_1 .- m_G1_0).^2)/sx_G2
    #s_G2_0 = sum((m_x_G2_0 .- m_G1_0).^2)/b
    #s_G2_1 = sum((m_x_G2_1 .- m_G1_0).^2)/sx_G2

    #s = s_G1_0_w, s_G1_1_w, s_G1_0, s_G1_1, s_G2_0_w, s_G2_1_w, s_G2_0, s_G2_1

    return m#, s
end

function f(a, b, c)

  #= a = Anzahl wie oft das Netz trainiert wird,
    b = Fallzahl fürs Training,
    c = Fallzahl fürs Auswerten =#
    d1 = zeros(2,6)
    s1 = zeros(1, 8)
    for i in 1:a
      println("Das ist die ", i, ".te Wiederholung, für n=", b)
      model, modelw = nn(b)
      x, y, x_G1, x_G2, xT = design(c)
      e = m(x, y, x_G1, x_G2, xT, model, modelw)
      for i in 1:2
        for j in 1:6
            d1[i, j] += e[i, j]
        end
      end
    end
    return d1/a
end

#weights = GewichtNN(10000)
 d10 = f(20, 1000, 100)
 #d1000 = f(5, 1000, 100)
 #d10000 = f(5, 10000, 100)
dxT0 = zeros(1, 6)
dxT0[1,:] = d10[1,:]
#dxT0[2,:] = d1000[1,:]
#dxT0[3,:] = d10000[1,:]

dxT1 = zeros(1, 6)
dxT1[1,:] = d10[2,:]
#dxT1[2,:] = d1000[2,:]
#dxT1[3,:] = d10000[2,:]


#ds = zeros(1, 8)
#ds[1,:] = d100[2,:]
#ds[2,:] = d1000[2,:]
#ds[3,:] = d10000[2,:]

dfm0 = convert(DataFrame, dxT0)
#rename!(dfm, f => t for (f, t) =
 #  zip([:x1, :x2, :x3, :x4, :x5, :x6],
  #     [:G1_0-G1_1, :wG1_0-G1_1, :dif, :G2_0-G2_1, :wG2_0-G2_1, dif]))
println("xT = 0", dfm0)

dfm1 = convert(DataFrame, dxT1)
#rename!(dfm, f => t for (f, t) =
 #  zip([:x1, :x2, :x3, :x4, :x5, :x6],
  #     [:G1_0-G1_1, :wG1_0-G1_1, :dif, :G2_0-G2_1, :wG2_0-G2_1, dif]))
println("XT = 1", dfm1)
#println("Gewichte mit 100.000 a = 50, epochen überall 500")

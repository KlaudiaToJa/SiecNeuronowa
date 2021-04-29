import pandas as pd
import numpy as np
import math

heart_df = pd.read_csv("dane.csv", sep=',')
print(heart_df.head())
#print(heart_df)

print("\nIlość danych:")
print(heart_df.shape)

# sprawdzanie braków:

print(heart_df.isna().sum())

# pierwotnie wystąpiło 201 braków w "bmi", całość obserwacji wynosi 5110, dlatego postanawiamy usunąć wiersze z brakami:

dane = heart_df.dropna(axis=0)

print(dane.shape)
print(dane.isna().sum())

# sprawdzamy typy występujące w danych:
print("Typy danych:")
print(dane.dtypes)

# tworzymy osobne kolumny dla kolumn, które zawierały więcej niż dwa rodzaje danych kategorycznych:

dane = dane.copy()

nazwy = ["children", "Private", "Self_employed", "Govt_job", "Never_worked"]
pom = []

for i in nazwy:
        for n in dane.loc[:, 'work_type']:
                if n == i:
                        pom.append(1)
                else:
                        pom.append(0)
        dane[i] = pom  # dodanie wektora jako nową kolumnę do danych
        pom = []

nazwy = ["Male", "Female", "Other"]
for i in nazwy:
        for n in dane.loc[:, 'gender']:
                if n == i:
                        pom.append(1)
                else:
                        pom.append(0)
        dane[i] = pom
        pom = []

for n in dane.loc[:, 'ever_married']:
        if n == "Yes":
                pom.append(1)
        else:
                pom.append(0)
dane['ever_married'] = pom
pom = []

nazwy = ["Rural", "Urban"]
for i in nazwy:
        for n in dane.loc[:, 'Residence_type']:
                if n == i:
                        pom.append(1)
                else:
                        pom.append(0)
        dane[i] = pom
        pom = []

nazwy = ['formerly_smoked', 'never_smoked', 'smokes']

for i in nazwy:
        for n in dane.loc[:, 'smoking_status']:
                if n == i:
                        pom.append(1)
                else:
                        pom.append(0)
        dane[i] = pom
        pom = []

dane.drop('gender', axis='columns', inplace=True)
dane.drop('id', axis='columns', inplace=True)
dane.drop('work_type', axis='columns', inplace=True)
dane.drop('smoking_status', axis='columns', inplace=True)
dane.drop('Residence_type', axis='columns', inplace=True)



# SIEĆ NEURONOWA:


# To, co dzieje się na złączeniach neuronów:
class Polaczenie:
    def __init__(self, polaczonyNeuron):
        self.polaczonyNeuron = polaczonyNeuron  # wskazuje, o którego neurona z aktualnej warstwy nam chodzi
        self.waga = np.random.normal()  # losuje wagę dla połączenia z tym neuronem
        self.deltaWag = 0.0


# Klasa samego neuronu, odpowiada za działania wewnątrz neuronów
class Neuron:
    eta = 0.01  # Learning Rate
    alfa = 0.9  # momentum, współczynnik wnoszący bezwładność. Przyspiesza i stabilizuje uczenie.
    

    def __init__(self, warstwa):
        self.dendrony = []
        self.blad = 0.0
        self.gradient = 0.0
        self.wynik = 0.0  # wynik z pojedynczego neuronu
        if warstwa is None:  # poprzednia warstwa; dla danych wejściowych przyjmuje wartość None 
            pass
        else:
            for neuron in warstwa:  # dla każdego pojedynczego neuronu w podanej poprzedniej warstwie
                pol = Polaczenie(neuron)  # utwórz połączenie
                self.dendrony.append(pol)  # i dodaj je do listy połączeń aktualnego neuronu w bieżącej warstwie

    def dodajBlad(self, bl):  # używane w propagacji wstecznej; zbiera informacje o błędach z każdego neuronu z poprzedniej warstwy
        self.blad = self.blad + bl

    def sigmoid(self, x):  # to jest funkcja aktywacyjna, tutaj sigmoidalna. Według wzoru: 1/(1+e^(-z)),
        # gdzie z = suma wag*wynik neuronów poprzednich + bias
        return 1 / (1 + np.exp(-x * 1.0))

    def dSigmoid(self, x):  # pochodna funkcji aktywacyjnej, używana do propagacji wstecznej.
        return x * (1.0 - x)

    def ustalBlad(self, bl):  # ustala wartość błędu
        self.blad = bl

    def ustalWynik(self, wynik):  # ustala wartość wyniku neuronu
        # potrzebne do forward propagation
        self.wynik = wynik

    def zwrocWynik(self):  # zwraca wartość wyniku
        return self.wynik

    def nauczGo(self): 
        sumaWynikow = 0
        if len(self.dendrony) == 0:  # jeżeli nie ma żadnego połączenia dla danego neurona, kończy działanie funkcji
            return
        for dendron in self.dendrony:
            sumaWynikow = sumaWynikow + float(dendron.polaczonyNeuron.zwrocWynik()) * dendron.waga  
        self.wynik = self.sigmoid(sumaWynikow)  

    def wstecznaPropagacja(self):
        self.gradient = self.blad * self.dSigmoid(self.wynik)
        for dendron in self.dendrony:
            dendron.deltaWag = Neuron.eta * self.gradient * dendron.polaczonyNeuron.wynik + self.alfa * dendron.deltaWag
            dendron.waga = dendron.waga + dendron.deltaWag
            dendron.polaczonyNeuron.dodajBlad(dendron.waga * self.gradient)
        self.blad = 0


# CAŁA ARCHITEKTURA SIECI:


class Siec:
    def __init__(self, architektura):
        self.warstwy = []
        for iloscNeuronow in architektura:  # dla każdej kolejnej warstwy w liście architektura…
            warstwa = []
            for i in range(iloscNeuronow):
                if len(self.warstwy) == 0:  # jeżeli jeszcze nie ma żadnej utworzonej warstwy, tworzony teraz neuron nie ma żadnej poprzedniej warstwy
                    warstwa.append(Neuron(None))
                else:
                    warstwa.append(Neuron(self.warstwy[-1]))
            if iloscNeuronow != 1:  # jeżeli nie jest neuronem w ostatniej warstwie... (w ostatniej nie potrzebujemy biasa)
                warstwa.append(Neuron(None))  # dodajemy bias jako ostatni neuron
                warstwa[-1].ustalWynik(1)  # wartość biasa wynosi 1
            self.warstwy.append(warstwa)  # dodajemy świeżo utworzoną warstwę do naszej kolekcji warstw

    def ustalWejscie(self, wejscia):  # ta metoda pozwala nam dodać do pierwszej warstwy nasze dane
        for i in range(len(wejscia)):
            self.warstwy[0][i].ustalWynik(wejscia[i])  # "0", bo w listach numerowanie jest od 0. "i" przechodzi po kolejnych wierszach = neuronach

    def liczBlad(self, cel):  # liczy błąd popełniony przez całą sieć (wszystkie warstwy) dla konkretnej pary wejście-wynik
        bl = 0
        e = (cel - self.warstwy[-1][0].zwrocWynik())  # oblicza różnicę dla podanego inputu pomiędzy wartością rzeczywistą a przewidywaną
        bl = bl + e ** 2  # err to będzie suma kwadratów powyższego działania dla każdej kolejnej pary input wynik
        bl = math.sqrt(bl)  # pierwiastek sumy błędów
        return bl

    def nauczGo(self):  # w tej funkcji wywołujemy nauczanie neuronów
        for warstwa in self.warstwy[1:]:  # dla każdej kolejnej warstwy idąc od pierwszej ukrytej
            for neuron in warstwa:  # i dla każdego neuronu w danej warstwie
                neuron.nauczGo()  # naucz go

    def wstecznaPropagacja(self, cel):  # wsteczna propagacja
        self.warstwy[-1][0].ustalBlad(cel - self.warstwy[-1][0].zwrocWynik())  # ustawienie błędu dla ostatniego neuronu wynikowego
        for warstwa in self.warstwy[::-1]:  # iteracja zaczynająca się od ostatniej warstwy
            for neuron in warstwa:
                neuron.wstecznaPropagacja()

    def zwrocRezultat(self):  # zwraca czysty wynik; jeszcze bez binaryzacji
        wynik = self.warstwy[-1][0].zwrocWynik()
        return wynik

    def zwrocZintWynik(
            self):  # zwraca wynik zbinaryzowany (dla funkcji sigmoidalnej >0.5 ustawia wynik na 1, w przeciwnym wypadku na 0)
        wynik = 0
        o = self.warstwy[-1][0].zwrocWynik()
        if o > 0.5:
            wynik = 1
        return wynik


# MAIN:
architektura = []

# ucinamy dane aby szybciej sprawdzić działanie
# w danych wejściowych jest tylko 209 przypadkow zawałów
danezawal = dane.loc[dane['stroke'] == 1]  
daneniezawal = dane.loc[dane['stroke'] == 0]

danezawal = danezawal.sample(n=40)
daneniezawal = daneniezawal.sample(n=40)

dane = daneniezawal
dane = dane.append(danezawal)

# ustalanie ilości neuronów w każdej kolejnej warstwie:
architektura.append(dane.shape[1] - 1)  # tyle neuronów w warstwie, ile jest kolumn w danych; odejmujemy 1, ponieważ liczymy bez stroke
architektura.append(dane.shape[1] - 1)
architektura.append(1) # warstwa wyjściowa

sc = Siec(architektura)  # tworzenie sieci z zadanymi parametrami

ograniczenieIteracji = 200

while True:
    if ograniczenieIteracji == 0:
        break
    bl = 0
    # losowanie kolejności
    losowane = dane.sample(frac=1).reset_index(drop=True)  # usuwanie ryzyka wpadnięcia w ekstremum lokalne

    wejscia = losowane.copy()
    wejscia.drop('stroke', axis='columns', inplace=True)

    wierszeDoListy = []
    for wiersz in wejscia.itertuples():
        my_list = [wiersz.age, wiersz.hypertension, wiersz.heart_disease, wiersz.ever_married, wiersz.avg_glucose_level, wiersz.bmi,
                   wiersz.children, wiersz.Private, wiersz.Self_employed, wiersz.Govt_job, wiersz.Never_worked, wiersz.Male, wiersz.Female,
                   wiersz.Other, wiersz.Rural, wiersz.Urban, wiersz.formerly_smoked, wiersz.never_smoked, wiersz.smokes]
        wierszeDoListy.append(my_list)

    wejscia = wierszeDoListy

    wyniki = losowane['stroke']
    wyniki.tolist()

    iloscWierszy = wyniki.count()

    for i in range(iloscWierszy):  # dla każdego kolejnego wiersza
        sc.ustalWejscie(wejscia[i])
        sc.nauczGo()
        sc.wstecznaPropagacja(wyniki[i])
        bl = bl + sc.liczBlad(wyniki[i])

    print("błąd: ", bl / iloscWierszy)
    print("średni błąd ma być maksymalnie: 0.3")
    if bl < iloscWierszy * 0.3:  # średni błąd przypadający na każdą obserwację uczącą ma być mniejszy niż 0.3
        break

    ograniczenieIteracji -= 1

while True:
    age = float(input("Podaj swój wiek:"))
    hypertension = float(input("Nadciśnienie: 1-tak; 0-nie"))
    heart_disease = int(input("Choroba serca: 1-tak; 0-nie"))
    ever_married = int(input("Czy kiedykolwiek miałeś/miałaś małżonka? 1-tak; 0-nie"))
    avg_glucose_level = float(input("Przeciętny poziom glukozy:"))
    bmi = float(input("BMI:"))
    children = int(input("Czy jesteś dzieckiem? 1-tak; 0-nie:"))
    if children == 0:
        private = int(input("Czy pracujesz w prywatnym sektorze? 1-tak; 0-nie:"))
        if private == 0:
            self_employed = int(input("Samozatrudnienie: 1-tak; 0-nie"))
            if self_employed == 0:
                govt_job = int(input("Czy pracujesz w publicznym sektorze: 1-tak; 0-nie"))
                if govt_job == 0:
                    never_worked = 1  # nigdy nie pracował
                else:
                    never_worked = 0  # już kiedyś pracował
            else:
                govt_job = 0
                never_worked = 0
        else:
            self_employed = 0
            govt_job = 0
            never_worked = 0
    else:
        private = 0
        self_employed = 0
        govt_job = 0
        never_worked = 0

    male = int(input("Jesteś mężczyzną? 1-tak; 0-nie"))
    if male == 1:
        female = 0
        other = 0
    else:
        female = int(input("Jesteś kobietą? 1-tak; 0-nie:"))
        if female == 0:
            other = 1  # other
        else:
            other = 0

    rural = int(input("Czy mieszkasz na obszarze wiejskim? 1-tak; 0-nie"))
    if rural == 1:
        urban = 0
    else:
        urban = 0

    smokes = int(input("Czy jesteś palaczem? 1-tak; 0-nie"))
    if smokes == 0:
        formerly_smoked = int(input("Czy jesteś byłym palaczem? 1-tak; 0-nie"))
        if formerly_smoked == 0:
            never_smoked = 1
        else:
            never_smoked = 0
    else:
        formerly_smoked = 0
        never_smoked = 0

    a = [age, hypertension, heart_disease, ever_married, avg_glucose_level, bmi, children, private,
         self_employed, govt_job, never_worked, male, female, other, rural, urban, formerly_smoked, never_smoked, smokes]

    sc.ustalWejscie(a)
    sc.nauczGo()

    print(sc.zwrocZintWynik())
    print(sc.zwrocRezultat())




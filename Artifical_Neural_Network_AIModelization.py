import numpy as np 

class YapayNoron:
    def __init__(self, ogrenme_orani=0.1, iterasyon_sayisi=1000):
        # Modelin hiperparametreleri
        self.ogrenme_orani = ogrenme_orani
        self.iterasyon_sayisi = iterasyon_sayisi
        self.w = None # Agirliklar (w1, w2... wn)
        self.b = None # Bias (Sapma)

    def _sigmoid(self, z):
        # Aktivasyon fonksiyonu: ��kt�y� 0 ile 1 aras�na s�k��t�r�r.
        return 1 / (1 + np.exp(-z))

    def egit(self, X, y):
        # X: Girdi verisi (�rnekler x �zellikler)
        # y: Ger�ek etiketler (0 veya 1)
        
        ornek_sayisi, ozellik_sayisi = X.shape

        # 1. Ba�lang�� (Initialization)
        # A��rl�klar� s�f�r vekt�r� olarak, bias'� 0 olarak ba�lat�yoruz.
        self.w = np.zeros(ozellik_sayisi)
        self.b = 0.0

        # 2. ��renme D�ng�s� (Training Loop)
        for _ in range(self.iterasyon_sayisi):
            
            # --- �LER� YAYILIM (FORWARD PROPAGATION) ---
            # Input -> Weighted Sum
            # z = w1*x1 + w2*x2 + ... + wn*xn + b
            z = np.dot(X, self.w) + self.b
            
            # Weighted Sum -> Activation -> Output
            a = self._sigmoid(z)

            # --- GER� YAYILIM VE ��RENME (BACKPROPAGATION) ---
            # Modelin ne kadar hata yapt���n� bul ve a��rl�klar� g�ncelle (Gradient Descent)
            # dw: A��rl�klar�n hataya g�re t�revi (y�n�)
            # db: Bias'�n hataya g�re t�revi
            dw = (1 / ornek_sayisi) * np.dot(X.T, (a - y))
            db = (1 / ornek_sayisi) * np.sum(a - y)

            # A��rl�k (w) ve Bias (b) G�ncellemesi
            self.w -= self.ogrenme_orani * dw
            self.b -= self.ogrenme_orani * db

    def tahmin_et(self, X):
        # ��renilmi� w ve b de�erleriyle yeni veri �zerinde tahmin yap
        z = np.dot(X, self.w) + self.b
        a = self._sigmoid(z)
        
        # Sigmoid ��kt�s� 0.5'ten b�y�kse 1, de�ilse 0 s�n�f�na ata
        tahminler = [1 if i > 0.5 else 0 for i in a]
        return np.array(tahminler)

import numpy as np
import pandas as pd
import os

class TensorRecommend:
    def __init__(self, k, lambda_, eta, data_entries, num_users, num_items, num_features, seed=42, loading=False,path=None):
        """
        k: latentna dimenzija
        lambda_: regularizacija
        eta: learning rate
        data_entries: lista tupleova (user_idx, item_idx, feature1_idx, ..., rating)
        num_users: broj korisnika
        num_items: broj itema
        num_features: dict {feature_name: broj_jedinstvenih_vrijednosti} ili lista brojeva
        """
        np.random.seed(seed)
        self.k = k
        self.lambda_ = lambda_
        self.eta = eta
        self.data_entries = data_entries
        self.item_features = {}
        for entry in self.data_entries:
            _, m_idx, *feature_indices, _ = entry
            if m_idx not in self.item_features:
                self.item_features[m_idx] = feature_indices

        # Ako je dict, uzmemo samo veličine
        if isinstance(num_features, dict):
            feature_sizes = list(num_features.values())
        else:
            feature_sizes = num_features

        self.num_features = len(feature_sizes)
        if loading:
            self.U = np.load(os.path.join(os.path.dirname(os.getcwd()),path,"U_matrix.npy"))
            self.M = np.load(os.path.join(os.path.dirname(os.getcwd()),path,"M_matrix.npy"))
            self.C = [np.load(os.path.join(os.path.dirname(os.getcwd()),path,f"C_matrix_{i}.npy")) for i in range(self.num_features)]
            self.S = np.load(os.path.join(os.path.dirname(os.getcwd()),path,"S_tensor.npy"))
        else:
            # Latentni faktori
            self.U = np.random.uniform(0,2, size=(num_users, k))
            self.M = np.random.uniform(0,2, size=(num_items, k))
            self.C = [np.random.uniform(0,2, size=(n, k)) for n in feature_sizes]
            # Tensor S dimenzija k x k x k x ... (2 + broj featurea)
            self.S = np.random.uniform(0,2, size=(k,) * (2 + self.num_features))

    # Predikcija za jedan unos (iz nekog razloga moras uvalit rating u tuple, ali se ne koristi)
    def predict(self, entry):
        """
        entry: tuple (u_idx, m_idx, feature1_idx, feature2_idx, ..., rating)
        """
        u_idx, m_idx, *feature_indices, _ = entry  
        vecs = [self.U[u_idx], self.M[m_idx]]
        vecs += [self.C[i][f_idx] for i, f_idx in enumerate(feature_indices)]

        n_dims = len(vecs)
        subscripts = ''.join([chr(ord('p') + i) for i in range(n_dims)])
        einsum_str = f"{subscripts},{','.join(subscripts)}->"
        return np.einsum(einsum_str, self.S, *vecs)

    # Računanje gubitka (loss_nr je samo MSE bez regularizacije, loss je ukupni gubitak s regularizacijom)
    def compute_loss(self):
        loss = 0.0
        for entry in self.data_entries:
            pred = self.predict(entry)
            r = entry[-1]
            loss += (r - pred) ** 2

        # L2 regularizacija
        reg = np.sum(self.U**2) + np.sum(self.M**2)
        for C_i in self.C:
            reg += np.sum(C_i**2)
        reg += np.sum(self.S**2)
        loss_nr = loss
        loss += self.lambda_ * reg
        return loss_nr, loss

    # Clipping gradijenta na max_norm
    def _clip_grad(self, grad, max_norm):
        norm = np.linalg.norm(grad)
        if norm > max_norm:
            grad = grad * (max_norm / norm)
        return grad

    # Trening za jednu epohu
    def train_epoch(self, t=1, initial_lr=0.01, max_norm=5.0):
        np.random.shuffle(self.data_entries)
        #u jednoj epohi prolazimo kroz sve unose i ažuriramo faktore i tensor S
        for entry in self.data_entries:
            u_idx, m_idx, *feature_indices, r = entry

            eta = initial_lr / np.sqrt(t)
            t += 1

            # --- Dohvati latentne vektore ---
            u_vec = self.U[u_idx]
            m_vec = self.M[m_idx]
            c_vecs = [self.C[i][f_idx] for i, f_idx in enumerate(feature_indices)]

            all_vecs = [u_vec, m_vec] + c_vecs
            n_modes = len(all_vecs)

            # --- Predikcija ---
            pred = self.predict(entry)
            err = pred - r
            #print("pred:", pred, "r:", r)


            # --- Gradijenti za faktore (U, M, C...) ---
            grads = []

            # Generiramo indekse a,b,c,d,...
            indices = [chr(ord('a') + i) for i in range(n_modes)]
            S_subs = ''.join(indices)

            for i in range(n_modes):
                # indeks koji ostaje (npr 'a')
                keep = indices[i]

                # svi ostali indeksi
                other_indices = [indices[j] for j in range(n_modes) if j != i]

                # vektori osim i-tog
                other_vecs = [all_vecs[j] for j in range(n_modes) if j != i]

                # npr: "abcd,b,c,d->a"
                einsum_str = f"{S_subs},{','.join(other_indices)}->{keep}"

                grad = err * np.einsum(einsum_str, self.S, *other_vecs)

                # L2 regularizacija
                grad += self.lambda_ * all_vecs[i]

                # clipping
                grad = self._clip_grad(grad, max_norm)


                grads.append(grad)
                #print("||grad_U||:", np.linalg.norm(grads[0]))

            # --- Gradijent za S ---
            grad_S = err * all_vecs[0]
            for v in all_vecs[1:]:
                grad_S = np.multiply.outer(grad_S, v)

            grad_S = self._clip_grad(grad_S, max_norm)

            # --- Update ---
            self.U[u_idx] -= eta * grads[0]
            self.M[m_idx] -= eta * grads[1]

            for i, f_idx in enumerate(feature_indices):
                self.C[i][f_idx] -= eta * grads[2 + i]

            self.S -= eta * grad_S

        loss_nr, loss = self.compute_loss()
        print(f"Epoch Loss: {loss:.4f}, NR Loss: {loss_nr:.4f}, avg error: {np.sqrt(loss_nr/len(self.data_entries)):.4f}")

        return t
    # Spremanje modela u .npy datoteke
    def save_model(self, path):
        np.save(os.path.join(path,"U_matrix.npy"), self.U)
        np.save(os.path.join(path,"M_matrix.npy"), self.M)
        for i, C_i in enumerate(self.C):
            np.save(os.path.join(path,f"C_matrix_{i}.npy"), C_i)
        np.save(os.path.join(path,"S_tensor.npy"), self.S)

    # Preporuka top N igara za danog korisnika (dani vektor korisnika)
    def top_games(self, user_vec, top_n=10):
        scores = []

        for m_idx in range(self.M.shape[0]):

            if m_idx not in self.item_features:
                continue

            feature_indices = self.item_features[m_idx]

            vecs = [user_vec, self.M[m_idx]]
            vecs += [self.C[i][f_idx] for i, f_idx in enumerate(feature_indices)]

            subscripts = ''.join([chr(ord('p') + i) for i in range(len(vecs))])
            einsum_str = f"{subscripts},{','.join(subscripts)}->"

            score = np.einsum(einsum_str, self.S, *vecs)

            scores.append((m_idx, score))

        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_n]


    # OVO SE MORA DOVRSITI, TREBA PROVJERITI DA LI RADI KAKO TREBA, JER JE KOMPLICIRANO
    def fit_new_user(self, user_ratings, epochs=50, lr=0.01, max_norm=5.0):
        """
        user_ratings: lista tupleova
        [(m_idx, f1_idx, f2_idx, ..., rating)]
        """

        u_vec = np.random.uniform(0, 2, size=self.k)

        for _ in range(epochs):
            for entry in user_ratings:
                m_idx, *feature_indices, r = entry

                vecs = [u_vec, self.M[m_idx]]
                vecs += [self.C[i][f_idx] for i, f_idx in enumerate(feature_indices)]

                # predikcija
                subscripts = ''.join([chr(ord('p') + i) for i in range(len(vecs))])
                einsum_str = f"{subscripts},{','.join(subscripts)}->"
                pred = np.einsum(einsum_str, self.S, *vecs)

                err = pred - r

                # gradijent za u_vec
                indices = [chr(ord('a') + i) for i in range(len(vecs))]
                S_subs = ''.join(indices)

                other_indices = indices[1:]
                einsum_str = f"{S_subs},{','.join(other_indices)}->a"

                grad_u = err * np.einsum(
                    einsum_str,
                    self.S,
                    *vecs[1:]
                )

                grad_u += self.lambda_ * u_vec
                grad_u = self._clip_grad(grad_u, max_norm)

                u_vec -= lr * grad_u

        return u_vec


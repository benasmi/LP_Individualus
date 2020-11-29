import math
import numpy as np
from scipy.spatial import distance
from matplotlib import pyplot as plt
from multiprocessing import Pool
from functools import partial
import time
import pickle

'''
Displays coordinates
'''


def show_graph(coords, extra_coords, required_circle):
    x, y = coords.T
    circle1 = plt.Circle((0, 0), required_circle, color='r', alpha=.3)
    plt.gcf().gca().add_artist(circle1)
    plt.xlim((-10, 10))
    plt.ylim((-10, 10))
    extra_x, extra_y = extra_coords.T
    plt.scatter(extra_x, extra_y)
    plt.scatter(x, y)
    plt.show()


'''
Calculates euclidean distance
'''


def euclidean_distance(p1, p2):
    return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)


'''
Objective funtion
'''


def objective_function(coords, extra_coords, average):
    extended = np.concatenate((coords, extra_coords), axis=0)

    from_center = 0
    for extra in extra_coords:
        from_center += (euclidean_distance(extra, [0, 0]) - from_start_required) ** 2
    from_center /= len(extra_coords)

    distance = 0
    for i in range(len(extended)):
        for j in range(len(extended)):
            if i != j:
                distance += (euclidean_distance(extended[i], extended[j]) - average[i]) ** 2
    return distance + from_center


'''
Get average and use it in optimization function
'''


def get_average_vector(coords):
    average = []
    for i in range(len(coords)):
        distance = 0
        for j in range(len(coords)):
            if i != j:
                distance += euclidean_distance(coords[i], coords[j])
        average.append(distance / (len(coords) - 1))
    return average


'''
Calculate gradient numerically
'''


def gradient(coords, extra_coords, average):
    h = 0.01
    n = extra_coords.shape
    nabla_xy = np.zeros(n)
    initial_loss = objective_function(coords, extra_coords, average)
    print(n[0])
    for i in range(0, n[0]):
        tmp_coords = np.array(extra_coords, copy=True)
        tmp_coords[i][0] += h
        loss = objective_function(coords, tmp_coords, average)
        nabla_xy[i][0] = (loss - initial_loss) / h
    for i in range(0, n[0]):
        tmp_coords = np.array(extra_coords, copy=True)
        tmp_coords[i][1] += h
        loss = objective_function(coords, tmp_coords, average)
        nabla_xy[i][1] = (loss - initial_loss) / h
    return nabla_xy


'''
Gradient - descent algorithm
'''


def gradient_pool(index, coords, extra_coords, average):
    h = 0.01  # Mažas delta poslinkis
    nabla_xy = np.empty(2, dtype=np.float32)  # Paruošiama vieta gradientui
    initial_loss = objective_function(coords, extra_coords, average)  # Pradinė tikslo funkcija

    # Dirbama su papildomo taško x ir y koordinatėmis
    for axis in [0, 1]:
        # Lokaliai nukopijuojame papildomas koordinates
        tmp_coords = np.array(extra_coords, copy=True)
        # Padidiname papildomą tašką per mažą delta poslinkį h
        tmp_coords[index][axis] += h
        # Paskaičiuojame tikslo funkcija su padidintu x arba y
        loss = objective_function(coords, tmp_coords, average)
        # Paskaičiuojame, kiek tas poslinkis turėjo įtakos ir išsaugome gradientą.
        nabla_xy[axis] = (loss - initial_loss) / h
    return nabla_xy


def optimize(p_count, coords, extra_coords, avg):
    step = 0.01  # Gradiento žingsnis.
    accuracy = 10e-10  # Reikalaujamas tikslumas.
    max_iterations = 5000
    losses = []  # Laikomi visi tikslo funkcijos rezultatai.
    idxs = list(range(len(extra_coords)))  # Papildomų taškų indeksų sąrašas

    # Atidaromas Pool objektas su pasirinktu procesų kiekiu.
    with Pool(processes=p_count) as pool:
        # Sukamas ciklas, tol kol pasiekiamas tikslumas arba iteracijų limitas
        for i in range(max_iterations):
            # Gaunama tikslo funkcijos reikšmė
            initial_loss = objective_function(coords=coords, extra_coords=extra_coords, average=avg)

            # Su 'partial' funkcija paruošiame gradiento funkciją pool.map metodui
            nabla_settings = partial(gradient_pool,
                                     coords=coords,
                                     extra_coords=extra_coords,
                                     average=avg)

            # Skaičiuojami gradientai kiekvienam taškui atskirai
            nabla_xy = np.array(pool.map(nabla_settings, idxs))

            # Pakeičiamos papildomos koordinatės pagal gautą gradientą
            extra_coords.transpose()[0] -= nabla_xy.transpose()[0] * step
            extra_coords.transpose()[1] -= nabla_xy.transpose()[1] * step
            new_loss = objective_function(coords=coords, extra_coords=extra_coords, average=avg)

            # Jeigu tikslo funkcija padidėjo(nes norime minimizuoti), reiškias per didelis žingnis
            # Reikia sugrįžti ir pamažinti žingsnį.
            if new_loss > initial_loss:
                extra_coords.transpose()[0] += nabla_xy.transpose()[0] * step
                extra_coords.transpose()[1] += nabla_xy.transpose()[1] * step
                step /= 2
            else:
                losses.append(new_loss)

            # Jeigu pasiektas norimas tikslumas arba maksimalus iteracijų skaičius -  nutraukiame optimizavimą
            if (abs(initial_loss - new_loss) / (initial_loss + new_loss)) < accuracy or i == max_iterations:
                return losses


from_start_required = 1


def show_objective_graph(results):
    plt.plot(results)
    plt.show()


def gen_coordinates(n):
    sx = np.random.randint(-10, 10, n).reshape(-1, 1).astype(np.float32)
    sy = np.random.randint(-10, 10, n).reshape(-1, 1).astype(np.float32)
    ex = np.random.randint(-10, 10, n).reshape(-1, 1).astype(np.float32)
    ey = np.random.randint(-10, 10, n).reshape(-1, 1).astype(np.float32)

    scoords = np.concatenate([sx, sy], axis=1)
    ecoords = np.concatenate([ex, ey], axis=1)
    all_coords = np.concatenate((scoords, ecoords), axis=0)

    return scoords, ecoords, all_coords


'''
Test optimization
'''


def launch(processors, n, verbose=True, scoords=None, ecoords=None, all_coords=None):
    if scoords is None or ecoords is None or all_coords is None:
        scoords, ecoords, all_coords = gen_coordinates(n)

    if verbose:
        print("Initial variance", np.var(all_coords))
        show_graph(scoords, ecoords, from_start_required)

    avg = get_average_vector(all_coords)
    start_time = time.time()
    losses = optimize(processors, scoords, ecoords, avg)
    end_time = time.time()
    total_time = end_time - start_time

    print(f'Time {total_time} with {processors} processors with data count {n}')
    if verbose:
        show_graph(scoords, ecoords, from_start_required)
        ext = np.concatenate((scoords, ecoords), axis=0)
        print("After optimization variance", np.var(ext))

        show_objective_graph(losses)
    return total_time, processors, n


def time_research():
    scoords, ecoords, _ = gen_coordinates(20)
    print("---START---")
    processes = [1,2,4,6,8,10,12,14,16,20]
    data = []
    for p in processes:
        print(f"---With process count: {p}---")
        tmp = []
        for j in range(5, 20):
            average = 0
            for idx in range(3):
                copy_scoords = np.array(scoords, copy=True)
                copy_ecoords = np.array(ecoords, copy=True)

                scoords_slice = copy_scoords[1:j]
                ecoords_slice = copy_ecoords[1:j]
                all_coords_slice =  np.concatenate((scoords_slice, ecoords_slice), axis=0)
                time_taken, _, _ = launch(processors=p,
                                          n=j,
                                          verbose=False,
                                          scoords=scoords_slice,
                                          ecoords=ecoords_slice,
                                          all_coords=all_coords_slice)
                average += time_taken

            print(f"----3 times average {average/3}  times | Processes: ({p}) <-> ({j})Data |----")
            tmp.append((average/3, p, j))
        data.append(tmp)
    print("---FINISH---")
    return data


if __name__ == '__main__':

    # # scoords, ecoords, all_coords = gen_coordinates(10)
    # scoords = np.array([[-10, 10], [0, 10], [10, 10], [10, 0], [10, -10], [0, -10], [-10, -10], [-10, 0]]).astype(np.float)
    # ecoords = np.array([[-10, 10], [0, 10], [10, 10], [10, 0], [10, -10], [0, -10], [-10, -10], [-10, 0]]).astype(np.float)
    # all_coords = np.concatenate((scoords, ecoords), axis=0)
    # launch(processors=1,n=10,verbose=True,scoords=scoords, ecoords=ecoords, all_coords=all_coords)
    # print("CHECK COORDS", scoords, ecoords, all_coords)
    # launch(processors=16,n=10,verbose=True,scoords=scoords, ecoords=ecoords, all_coords=all_coords)
    results = time_research()
    print(results)

    with open('outfile.txt', 'wb') as fp:
       pickle.dump(results, fp)

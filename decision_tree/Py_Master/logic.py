import random
import constants as c

def new_game(n):
    matrix = [[0]*n for _ in range(n)]
    matrix = add_two(matrix)
    matrix = add_two(matrix)
    return matrix

def add_two(mat):
    empty_cells = [(i, j) for i in range(len(mat)) for j in range(len(mat[0])) if mat[i][j]==0]
    if empty_cells:
        i, j = random.choice(empty_cells)
        mat[i][j] = 2
    return mat

def game_state(mat):
    size = len(mat)
    for i in range(size):
        for j in range(size):
            if mat[i][j] == 2048:
                return 'win'
            if mat[i][j] == 0:
                return 'not over'
            if (i < size - 1 and mat[i][j] == mat[i+1][j]) or (j < size - 1 and mat[i][j] == mat[i][j+1]):
                return 'not over'
    return 'lose'

def reverse(mat):
    return [row[::-1] for row in mat]

def transpose(mat):
    return [list(row) for row in zip(*mat)]

def tighten(row):
    '''Squash non-zero elements together'''
    new_row = [i for i in row if i != 0]
    return new_row + [0]*(len(row) - len(new_row))

def merge(row):
    '''Merge row elements'''
    for i in range(len(row)-1):
        if row[i] == row[i+1] and row[i] != 0:
            row[i] *= 2
            row[i+1] = 0
    return row

def left(mat):
    done = False
    new_mat = []
    for i in range(len(mat)):
        row = mat[i]
        tight_row = tighten(row)
        merged_row = merge(tight_row)
        tight_merged_row = tighten(merged_row)
        if tight_merged_row != mat[i]:
            done = True
        new_mat.append(tight_merged_row)
    return new_mat, done

def right(mat):
    done = False
    new_mat = []
    for i in range(len(mat)):
        row = mat[i][::-1]
        tight_row = tighten(row)
        merged_row = merge(tight_row)
        tight_merged_row = tighten(merged_row)
        final_row = tight_merged_row[::-1]
        if final_row != mat[i]:
            done = True
        new_mat.append(final_row)
    return new_mat, done

def up(mat):
    transposed = transpose(mat)
    new_mat, done = left(transposed)
    return transpose(new_mat), done

def down(mat):
    transposed = transpose(mat)
    new_mat, done = right(transposed)
    return transpose(new_mat), done
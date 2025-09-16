import base64
import io
import os
import pdfkit
import streamlit as st
import streamlit.components.v1 as components
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
import random
import re
import seaborn as sns
import string
import textwrap
import time
import tqdm

from copy import copy as duplicate
from fpdf import FPDF

class Crossword(object):
    def __init__(self, cols, rows, empty='-', maxloops=4000, available_words=[]):
        self.cols = cols
        self.rows = rows
        self.empty = empty
        self.maxloops = maxloops
        self.available_words = available_words
        self.randomize_word_list()
        self.current_word_list = []
        self.clear_grid()
        self.debug = 0
 
    def clear_grid(self):
        """Initialize grid and fill with empty character."""
        self.grid = []
        for i in range(self.rows):
            ea_row = []
            for j in range(self.cols):
                ea_row.append(self.empty)
            self.grid.append(ea_row)
 
    def randomize_word_list(self):
        """Reset words and sort by length."""
        temp_list = []
        for word in self.available_words:
            if isinstance(word, Word):
                temp_list.append(Word(word.word, word.clue))
            else:
                temp_list.append(Word(word[0], word[1]))
        # randomize word list
        random.shuffle(temp_list)
        # sort by length
        temp_list.sort(key=lambda i: len(i.word), reverse=True)
        self.available_words = temp_list
 
    def compute_crossword(self, time_permitted=1.50, spins=3):
        copy = Crossword(self.cols, self.rows, self.empty,
                         self.maxloops, self.available_words)

        count = 0
        time_permitted = float(time_permitted)
        start_full = float(time.time())

        # only run for x seconds
        while (float(time.time()) - start_full) < time_permitted or count == 0:
            self.debug += 1
            copy.randomize_word_list()
            copy.current_word_list = []
            copy.clear_grid()

            x = 0
            # spins; 2 seems to be plenty
            while x < spins:
                for word in copy.available_words:
                    if word not in copy.current_word_list:
                        copy.fit_and_add(word)
                x += 1
            #print(copy.solution())
            #print(len(copy.current_word_list), len(self.current_word_list), self.debug)
            # buffer the best crossword by comparing placed words
            if len(copy.current_word_list) > len(self.current_word_list):
                self.current_word_list = copy.current_word_list
                self.grid = copy.grid
            count += 1
        return
 
    def suggest_coord(self, word):
        #count = 0
        coordlist = []
        glc = -1

        # cycle through letters in word
        for given_letter in word.word:
            glc += 1
            rowc = 0
            # cycle through rows
            for row in self.grid:
                rowc += 1
                colc = 0
                # cycle through letters in rows
                for cell in row:
                    colc += 1
                    # check match letter in word to letters in row
                    if given_letter == cell:
                        # suggest vertical placement 
                        try:
                            # make sure we're not suggesting a starting point off the grid
                            if rowc - glc > 0:
                                # make sure word doesn't go off of grid
                                if ((rowc - glc) + word.length) <= self.rows:
                                    coordlist.append([colc, rowc-glc, 1, colc+(rowc-glc),0])
                        except:
                            pass

                        # suggest horizontal placement 
                        try:
                            # make sure we're not suggesting a starting point off the grid
                            if colc - glc > 0: 
                                # make sure word doesn't go off of grid
                                if ((colc - glc) + word.length) <= self.cols:
                                    coordlist.append([colc-glc, rowc, 0, rowc+(colc-glc),0])
                        except:
                            pass

        # example: coordlist[0] = [col, row, vertical, col + row, score]
        #print(word.word)
        #print(coordlist)
        new_coordlist = self.sort_coordlist(coordlist, word)
        #print(new_coordlist)

        return new_coordlist
 
    def sort_coordlist(self, coordlist, word):
        """Give each coordinate a score, then sort."""
        new_coordlist = []
        for coord in coordlist:
            col, row, vertical = coord[0], coord[1], coord[2]
            # checking scores
            coord[4] = self.check_fit_score(col, row, vertical, word)
            # 0 scores are filtered
            if coord[4]:
                new_coordlist.append(coord)
        # randomize coord list; why not?
        random.shuffle(new_coordlist)
        # put the best scores first
        new_coordlist.sort(key=lambda i: i[4], reverse=True)
        return new_coordlist
 
    def fit_and_add(self, word):
        """Doesn't really check fit except for the first word;
        otherwise just adds if score is good.
        """
        fit = False
        count = 0
        coordlist = self.suggest_coord(word)
 
        while not fit and count < self.maxloops:
            # this is the first word: the seed
            if len(self.current_word_list) == 0:
                # top left seed of longest word yields best results (maybe override)
                vertical, col, row = random.randrange(0, 2), 1, 1

                if self.check_fit_score(col, row, vertical, word):
                    fit = True
                    self.set_word(col, row, vertical, word, force=True)

            # a subsquent words have scores calculated
            else:
                try:
                    col, row, vertical = coordlist[count][0], coordlist[count][1], coordlist[count][2]
                # no more cordinates, stop trying to fit
                except IndexError:
                    return
 
                # already filtered these out, but double check
                if coordlist[count][4]:
                    fit = True
                    self.set_word(col, row, vertical, word, force=True)
 
            count += 1

        return
 
    def check_fit_score(self, col, row, vertical, word):
        """Return score: 0 signifies no fit, 1 means a fit, 2+ means a cross.
        The more crosses the better.
        """
        if col < 1 or row < 1:
            return 0

        # give score a standard value of 1, will override with 0 if collisions detected
        count, score = 1, 1
        for letter in word.word:
            try:
                active_cell = self.get_cell(col, row)
            except IndexError:
                return 0
 
            if active_cell == self.empty or active_cell == letter:
                pass
            else:
                return 0
 
            if active_cell == letter:
                score += 1
 
            if vertical:
                # check surroundings
                if active_cell != letter: # don't check surroundings if cross point
                    if not self.check_if_cell_clear(col+1, row): # check right cell
                        return 0
 
                    if not self.check_if_cell_clear(col-1, row): # check left cell
                        return 0
 
                if count == 1: # check top cell only on first letter
                    if not self.check_if_cell_clear(col, row-1):
                        return 0
 
                if count == len(word.word): # check bottom cell only on last letter
                    if not self.check_if_cell_clear(col, row+1): 
                        return 0
            else: # else horizontal
                # check surroundings
                if active_cell != letter: # don't check surroundings if cross point
                    if not self.check_if_cell_clear(col, row-1): # check top cell
                        return 0
 
                    if not self.check_if_cell_clear(col, row+1): # check bottom cell
                        return 0
 
                if count == 1: # check left cell only on first letter
                    if not self.check_if_cell_clear(col-1, row):
                        return 0
 
                if count == len(word.word): # check right cell only on last letter
                    if not self.check_if_cell_clear(col+1, row):
                        return 0
 
            if vertical: # progress to next letter and position
                row += 1
            else: # else horizontal
                col += 1
 
            count += 1
 
        return score
 
    def set_word(self, col, row, vertical, word, force=False):
        """Set word in the grid, and adds word to word list."""
        if force:
            word.col = col
            word.row = row
            word.vertical = vertical
            self.current_word_list.append(word)
 
            for letter in word.word:
                self.set_cell(col, row, letter)
                if vertical:
                    row += 1
                else:
                    col += 1

        return
 
    def set_cell(self, col, row, value):
        self.grid[row-1][col-1] = value
 
    def get_cell(self, col, row):
        return self.grid[row-1][col-1]
 
    def check_if_cell_clear(self, col, row):
        try:
            cell = self.get_cell(col, row)
            if cell == self.empty: 
                return True
        except IndexError:
            pass
        return False
 
    def solution(self):
        """Return solution grid."""
        outStr = ""
        for r in range(self.rows):
            for c in self.grid[r]:
                outStr += '%s ' % c
            outStr += '\n'
        return outStr
 
    def word_find(self):
        """Return solution grid."""
        outStr = ""
        for r in range(self.rows):
            for c in self.grid[r]:
                if c == self.empty:
                    outStr += '%s ' % string.ascii_lowercase[random.randint(0,len(string.ascii_lowercase)-1)]
                else:
                    outStr += '%s ' % c
            outStr += '\n'
        return outStr
 
    def order_number_words(self):
        """Orders words and applies numbering system to them."""
        self.current_word_list.sort(key=lambda i: (i.col + i.row))
        count, icount = 1, 1
        for word in self.current_word_list:
            word.number = count
            if icount < len(self.current_word_list):
                if word.col == self.current_word_list[icount].col and word.row == self.current_word_list[icount].row:
                    pass
                else:
                    count += 1
            icount += 1
 
    def display(self, order=True):
        """Return (and order/number wordlist) the grid minus the words adding the numbers"""
        outStr = ""
        if order:
            self.order_number_words()
 
        copy = self
 
        for word in self.current_word_list:
            copy.set_cell(word.col, word.row, word.number)
 
        for r in range(copy.rows):
            for c in copy.grid[r]:
                outStr += '%s ' % c
            outStr += '\n'
 
        outStr = re.sub(r'[a-z]', '-', outStr)
        return outStr
 
    def word_bank(self):
        outStr = ''
        temp_list = duplicate(self.current_word_list)
        # randomize word list
        random.shuffle(temp_list)
        for word in temp_list:
            outStr += '%s\n' % word.word
        return outStr
 
    def legend(self):
        """Must order first."""
        outStr = 'Across:\n'
        for word in self.current_word_list:
            if word.down_across()=='across':
                space_pad = '   ' if word.number <10 else ' '
                clue = '\n      '.join(textwrap.TextWrapper(width=60).wrap(word.clue))
                outStr += f'{word.number:.0f}.{space_pad}{clue}\n'

        outStr += '\nDown:\n'
        for word in self.current_word_list:
            if word.down_across()=='down':
                space_pad = '   ' if word.number <10 else ' '
                clue = '\n      '.join(textwrap.TextWrapper(width=60).wrap(word.clue))
                outStr += f'{word.number:.0f}.{space_pad}{clue}\n'
        return outStr


class Word(object):
    def __init__(self, word=None, clue=None):
        self.word = re.sub(r'\s', '', word.lower())
        self.clue = clue
        self.length = len(self.word)
        # the below are set when placed on board
        self.row = None
        self.col = None
        self.vertical = None
        self.number = None
 
    def down_across(self):
        """Return down or across."""
        if self.vertical: 
            return 'down'
        else: 
            return 'across'
 
    def __repr__(self):
        return self.word

def create_pdf(img_fn, pdf_fn):
    """
    Create pdf written to pdf_fn with the image file img_fn.
    """
    pdf = FPDF()
    pdf.add_page()

    # Save to pdf
    pdf.set_xy(30, 50)
    pdf.image(img_fn, w=140, h=110)
    pdf.output(pdf_fn)


word_list = [list(x) for x in pd.read_csv('https://docs.google.com/spreadsheets/d/1Cq4oKuEy70fy31rYfRakSLjU4YmoZMfS6YxnbpaC0lk/export?format=csv').to_numpy()]

def generate_crossword(word_list):
    size = 12
    spins = 3
    maxloops = 5000
    
    sizes_checked = 1/14
    my_bar = st.progress(sizes_checked, text=f'Trying {size}x{size} grid')
    a = Crossword(size, size, '_', maxloops, word_list)
    a.compute_crossword(spins)
    while (len(a.current_word_list) != len(word_list)) & (size < 26):
        size +=1 
        sizes_checked += 1/14
        my_bar.progress(sizes_checked, text=f'Trying {size}x{size} grid')
        a = Crossword(size, size, '_', maxloops, word_list)
        a.compute_crossword(spins)
        
    my_bar.empty()
    return a, size

def plot_crossword(a, size, checkbox):
    letter_lists = [x.split() for x in a.display().split('\n')[:-1]]
    word_df = pd.DataFrame(letter_lists).replace('_',None)
    if (len(word_df[size-1].unique())==1) & (len(word_df.iloc[size-1].unique())==1):
        word_df = word_df.dropna(axis=0,how='all').dropna(axis=1,how='all')
    word_df.insert(0, 'first', [None] * word_df.shape[0])
    word_df.insert(word_df.shape[1], 'last', [None] * (word_df.shape[0]))
    word_df = pd.concat([pd.DataFrame(columns=word_df.columns,
                                      data=[[None] * word_df.shape[1]]),
                        word_df,
                        pd.DataFrame(columns=word_df.columns,
                                      data=[[None] * word_df.shape[1]])],
                        ignore_index=True)

    page_scales = len(letter_lists)*0.02
    fig, axs = plt.subplots(word_df.shape[0]+1, word_df.shape[1],
                            figsize=(17,22),
                            sharex='row', sharey='row',
                            dpi=600,
                            gridspec_kw={'hspace':0,'wspace':0,
                                        'height_ratios':[(1-page_scales)/word_df.shape[0]]*word_df.shape[0]+[page_scales]})
    for x in range(word_df.shape[1]):
        for y in range(word_df.shape[0]):
            if letter_lists[y-1][x-1] != '_':
                axs[y,x].set_xticks([])
                axs[y,x].set_yticks([])
                if letter_lists[y-1][x-1] != '-':
                    axs[y,x].text(0.05,0.95,letter_lists[y-1][x-1],ha='left',va='top')
            else:
                axs[y,x].set_axis_off()
        axs[word_df.shape[0],x].set_axis_off()
    axs[word_df.shape[0],0].text(1,0.95,a.legend(),va='top',fontsize=18)
    axs[word_df.shape[0],0].set_axis_off()
    if checkbox:
        axs[word_df.shape[0],word_df.shape[1]-3].text(1,0.95,'Word Bank:\n\n'+a.word_bank(),va='top',fontsize=18)
        axs[word_df.shape[0],word_df.shape[1]-3].set_axis_off()
    sns.despine(top=False,right=False)
    pdf_name = 'crossword.pdf'
    fig.savefig(pdf_name,format='pdf')
    # create_pdf(img_name, pdf_name)

    with open(pdf_name, 'rb') as h_pdf:
        st.download_button(
            label="Download PDF",
            data=h_pdf,
            file_name=pdf_name,
            mime="application/pdf",
            icon=":material/download:",
        )
    st.pyplot(fig)

st.set_page_config(page_title='Word Puzzle Generator', page_icon='https://static.nytimes.com/assets-oma/images/crossword-icon.svg',layout="wide")
st.title('Word Puzzle Generator')
st.write('Pulls words and definitions from [this Google Sheet](https://docs.google.com/spreadsheets/d/1Cq4oKuEy70fy31rYfRakSLjU4YmoZMfS6YxnbpaC0lk/edit?usp=sharing)')

st.header('Crossword Puzzle')
checkbox = st.checkbox('Include word bank', value='')
if st.button('Generate and preview Crossword Puzzle from word list'):
    a, size = generate_crossword(word_list)
    if len(a.current_word_list) != len(word_list):
        st.write('Could not fit all words into a crossword')
    else:
        plot_crossword(a, size,checkbox)

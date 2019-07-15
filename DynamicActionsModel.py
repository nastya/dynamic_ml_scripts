#!/usr/bin/python2
import sys
import json
import difflib
from sets import Set
import numpy as np
import gc
import datetime
import re
import os

flag_remove_cmdline = True

def compare_elem(elem1, elem2):
	global flag_remove_cmdline
	if elem1["type"] != elem2["type"]:
		return False
	else:
		if elem1["type"] == "DexClassLoader":
			return elem1["detail"]["path"] == elem2["detail"]["path"]
		if elem1["type"] in ["FileRW", "FdAccess"]:
			if not flag_remove_cmdline:
				cmdline_elem1_flag = (re.match(r"/proc/[0-9]+/cmdline", elem1["detail"]["path"]) != None)
				cmdline_elem2_flag = (re.match(r"/proc/[0-9]+/cmdline", elem2["detail"]["path"]) != None)
				if cmdline_elem1_flag != cmdline_elem2_flag:
					return False
				if cmdline_elem1_flag and elem1["type"] == "FdAccess":
					return True
			#NOTE: possible enhancement is finding better way to compare accessed files,
			#get files in the process of execution and compare the content
			if elem1["detail"]["path"][(elem1["detail"]["path"].rfind('/') + 1) :] != elem2["detail"]["path"][(elem2["detail"]["path"].rfind('/') + 1) :]:
				return False
		if elem1["type"] == "FileRW":
			#NOTE need to implement imprecise comparison of data (after checking the results)
			if elem1["detail"]["operation"] != elem2["detail"]["operation"] or elem1["detail"]["data"] != elem2["detail"]["data"]:
				return False
		return True

def longestCommonSubsequence(api_chain1, api_chain2, lcs = None):
	if (len(api_chain1) == 0 or len(api_chain2) == 0):
		return 0

	f = [[0 for x in range(len(api_chain2) + 1)] for x in range(len(api_chain1) + 1)]
	prev = [[(0,0) for x in range(len(api_chain2) + 1)] for x in range(len(api_chain1) + 1)]
	for i in range(0, len(api_chain1) + 1):
		for j in range(0, len(api_chain2) + 1):
			if (i == 0 or j == 0):
				f[i][j] = 0
			elif compare_elem(api_chain1[i - 1], api_chain2[j - 1]):
				f[i][j] = f[i-1][j-1] + 1
				prev[i][j] = (i-1, j-1)
			else:
				if (f[i-1][j] > f[i][j-1]):
					f[i][j] = f[i-1][j]
					prev[i][j] = (i-1, j)
				else:
					f[i][j] = f[i][j-1]
					prev[i][j] = (i, j-1)
	if lcs == None:
		return f[len(api_chain1) ][len(api_chain2)]

	#getting the longestCommonSubsequence itself
	i = len(api_chain1)
	j = len(api_chain2)

	while i > 0 and j > 0:
		i_prev, j_prev = prev[i][j]
		if i_prev != i and j_prev != j:
			lcs.insert(0, api_chain1[i_prev])
		i = i_prev
		j = j_prev

	return f[len(api_chain1)][len(api_chain2)]

class AppModel:
	def __init__(self, dirname = None):
		self.sequences = {}
		self.dirname = dirname
		self.filename = dirname + '/d_model.json' if dirname != None else None
		self.pkg_name = ''
		for f in os.listdir(dirname):
			if 'dumpsys_' in f:
				self.pkg_name = f[f.rfind("_") + 1 : f.rfind(".txt")]
				break
		self.json_model = {}
		self.chains = []
		if self.filename != None:
			self.build_by_file(self.filename)
		self.concatFileRW()

	def length(self):
		return len(self.chains)

	def avg_chain_length(self):
		sum_len = 0
		for ch in self.chains:
			sum_len += len(ch)
		return sum_len/len(self.chains) if len(self.chains) != 0 else 0

	def max_chain_length(self):
		max_len = 0
		for ch in self.chains:
			if len(ch) > max_len:
				max_len = len(ch)
		return max_len

	def compare(self, other_model):
		lcs_len = 0
		best_rel_val = 0
		best_match = []
		for chain1 in self.chains:
			for chain2 in other_model.chains:
				lcs = []
				length = longestCommonSubsequence(chain1, chain2, lcs)
				if length > lcs_len:
					lcs_len = length
					best_match = lcs
					best_rel_val = max(best_rel_val, length * 1.0 /len(chain2))
		return lcs_len, best_rel_val # best_match

	def removeUnrelatedActions(self, chain):
		i = 0
		while i < len(chain):
			elem = chain[i]
			if self.pkg_name != None  and not self.pkg_name in elem['process']:
				del chain[i]
				continue
			if elem["type"] == "FdAccess": #omitting FdAccess temporarily, too many of such entries
				del chain[i]
				continue
			if elem["type"] in ["FileRW", "FdAccess"] and (re.match(r"/proc/[0-9]+/cmdline", elem["detail"]["path"]) != None):
				del chain[i]
				continue
			if elem["type"] == "DexClassLoader" and "/system/framework" in elem["detail"]["path"]:
				del chain[i]
				continue
			i += 1
		return chain

	def concatFileRWChain(self, chain):
		new_chain = []
		i = 0
		while i < len(chain):
			elem = chain[i]
			if elem["type"] == "FileRW":
				op = elem["detail"]["operation"]
				path = elem["detail"]["path"]
				j = i
				data = elem["detail"]["data"]
				while j + 1 < len(chain) and chain[j+1]["type"] == "FileRW" and chain[j+1]["detail"]["operation"] == op and \
						chain[j+1]["detail"]["path"] == path:
					data += chain[j+1]["detail"]["data"]
					j = j + 1
				elem["detail"]["data"] = data
				new_chain.append(elem)
				i = j
			else:
				new_chain.append(elem)
			i = i + 1
		return new_chain

	def concatFileRW(self):
		new_chains = []
		for chain in self.chains:
			new_chains.append(self.concatFileRWChain(chain))
		self.chains = new_chains


	def build_chains(self, root_node, nodes_gone):
		if not root_node in self.transformed_model:
			#chain is built
			nodes_gone.append(root_node)
			#print 'Chain nodes:', nodes_gone #chain_until_now

			chain_until_now_built = []
			for i in range(0, len(nodes_gone) - 1):
				#unlooping self-edges
				for ev in self.transformed_model[nodes_gone[i]]:
					if self.transformed_model[nodes_gone[i]][ev]['endState'] == nodes_gone[i]:
						chain_until_now_built += self.transformed_model[nodes_gone[i]][ev]['reaction']
						break
				for ev in self.transformed_model[nodes_gone[i]]:
					if self.transformed_model[nodes_gone[i]][ev]['endState'] == nodes_gone[i + 1]:
						chain_until_now_built += self.transformed_model[nodes_gone[i]][ev]['reaction']
						break
			if len(self.removeUnrelatedActions(chain_until_now_built)) != 0 and not self.already_added_chain(self.removeUnrelatedActions(chain_until_now_built)):
				self.chains.append(self.removeUnrelatedActions(chain_until_now_built))
			return
		ending_flag = True
		for ev in self.transformed_model[root_node]:
			if self.transformed_model[root_node][ev]['endState'] == root_node:
				continue
			if self.transformed_model[root_node][ev]['endState'] in nodes_gone:
				continue
			#need to group here by the end state just like the self-edges
			new_nodes_gone = nodes_gone[:]
			new_nodes_gone.append(root_node)
			self.build_chains(self.transformed_model[root_node][ev]['endState'], new_nodes_gone)
			ending_flag = False
		if ending_flag:
			nodes_gone.append(root_node)
			chain_until_now_built = []
			for i in range(0, len(nodes_gone) - 1):
				#unlooping self-edges
				for ev in self.transformed_model[nodes_gone[i]]:
					if self.transformed_model[nodes_gone[i]][ev]['endState'] == nodes_gone[i]:
						chain_until_now_built += self.transformed_model[nodes_gone[i]][ev]['reaction']
						break
				for ev in self.transformed_model[nodes_gone[i]]:
					if self.transformed_model[nodes_gone[i]][ev]['endState'] == nodes_gone[i + 1]:
						chain_until_now_built += self.transformed_model[nodes_gone[i]][ev]['reaction']
						break
			if len(self.removeUnrelatedActions(chain_until_now_built)) != 0 and not self.already_added_chain(self.removeUnrelatedActions(chain_until_now_built)):
				self.chains.append(self.removeUnrelatedActions(chain_until_now_built))

	def already_added_chain(self, chain):
		for j in range(len(self.chains)):
			ch = self.chains[j]
			res = True
			for i in range(min(len(ch), len(chain))):
				if cmp(ch[i], chain[i]): #elems are UNequal
					res = False
					break
			if res and len(chain) > len(ch): #enlarge incomplete chain
				for i in range(min(len(ch), len(chain)), len(chain)):
					self.chains[j].append(chain[i])
			if res:
				return True
		return False

	def transform_model(self):
		self.transformed_model = {}
		for state in self.json_model:
			dict_flattened = {}
			for ev in self.json_model[state]:
				end_state = self.json_model[state][ev]['endState']
				if not end_state in dict_flattened:
					dict_flattened[end_state] = {}
					dict_flattened[end_state]["event"] = ''
					dict_flattened[end_state]["reaction"] = []
				dict_flattened[end_state]["event"] += ev + ', '
				dict_flattened[end_state]["reaction"] += self.json_model[state][ev]['reaction']
			self.transformed_model[state] = {}
			for st in dict_flattened:
				self.transformed_model[state][dict_flattened[st]["event"]] = {}
				self.transformed_model[state][dict_flattened[st]["event"]]["endState"] = st
				self.transformed_model[state][dict_flattened[st]["event"]]["reaction"] = dict_flattened[st]["reaction"]

	def build_from_json(self):
		#find the state without incoming edges
		in_states = []
		for state in self.json_model:
			for ev in self.json_model[state]:
				if not self.json_model[state][ev]['endState'] == state and \
					not self.json_model[state][ev]['endState'] in in_states:
					in_states.append(self.json_model[state][ev]['endState'])
		self.transform_model()
		start_states = []
		for state in self.json_model:
			if not state in in_states:
				start_states.append(state)
		if start_states == []:
			ts_dt_first = datetime.datetime.now()
			ts_int_first = -1
			first_st = ''
			if start_states == []:
				for state in self.json_model:
					for ev in self.json_model[state]:
						ts = ev[ : ev.find('~')]
						try:
							ts_dt = datetime.datetime.strptime(ts, "%Y-%m-%d_%H%M%S")
							if ts_dt < ts_dt_first:
								ts_dt_first = ts_dt
								first_st = state
						except ValueError:
							if ts_int_first == -1:
								ts_int_first = int(ts)
								first_st = state
							if int(ts) < ts_int_first:
								ts_int_first = int(ts)
								first_st = state
			if first_st != '':
				start_states.append(first_st)
		for st in start_states:
			self.build_chains(st, [])
		self.transformed_model = {}
		self.json_model = {}
		gc.collect()

	def build_by_file(self, filename):
		try:
			self.json_model = json.loads(open(filename, 'r').read())
			#Decoding the encoded data (read or written)
			for state in self.json_model:
				for event in self.json_model[state]:
					for elem in self.json_model[state][event]["reaction"]:
						if "detail" in elem and "data" in elem["detail"]:
							try:
								decoded = elem["detail"]["data"].decode("hex")
								elem["detail"]["data"] = decoded
							except TypeError:
								continue
			self.build_from_json()
		except ValueError as e:
			print 'ValueError happened', e
			return None

	def print_model(self):
		for chain in self.chains:
			print 'Chain:',
			for elem in chain:
					print elem,
			print ''

 



def sync_id_and_name(path, files):
	def add(k, v, dic):
		if k not in dic:
			dic[k] = v
		elif dic[k] != v:
			return False
		return True
	
	id_to_ent = {}
	ent_to_id = {}
	id_to_rel = {}
	rel_to_id = {}
	
	for file in files:
		id_file = path+'/'+file
		str_file = path+'/'+file+'.txt'
		id_lines = []
		str_lines = []
		print(id_file, str_file)
		with open(id_file,'r') as f:
			for line in f:
				line = line.strip('\n')
				ids = [int(id) for id in line.split('\t')]
				id_lines.append(ids)
		with open(str_file,'r',encoding='utf-8') as f:
			for line in f:
				line = line.strip('\n')
				strs = [s for s in line.split('\t')]
				str_lines.append(strs)
		for ids,str_line in zip(id_lines, str_lines):
			# print(ids, str_line)
			if add(str_line[1], ids[1], rel_to_id) is False:
				print("rel to id error")
			if add(ids[1],str_line[1], id_to_rel) is False:
				print("id to rel error")
	
			if add(str_line[0], ids[0], ent_to_id) is False:
				print(str_line[0],ids[0], ent_to_id[str_line[0]])
				print("ent to id error")
			if add(ids[0],str_line[0], id_to_ent) is False:
				print("id to ent error")
				
			if add(str_line[2], ids[2], ent_to_id) is False:
				print(str_line[2],ids[2], ent_to_id[str_line[2]])
				print("ent to id error")
			if add(ids[2],str_line[2], id_to_ent) is False:
				print("id to ent error")
	
	with open(path+'/'+'ent.dict','a',encoding='utf-8') as f:
		for k,v in id_to_ent.items():
			f.write(str(k)+'\t'+v+'\n')
	
	with open(path+'/'+'rel.dict','a',encoding='utf-8') as f:
		for k,v in id_to_rel.items():
			f.write(str(k)+'\t'+v+'\n')
	return id_to_ent, ent_to_id, id_to_rel, rel_to_id
			
def load_dict(path):
	id_to_b = {}
	b_to_ib = {}
	with open(path,'r',encoding='utf-8') as f:
		for line in f:
			line = line.strip('\n')
			s = line.split('\t')
			id_to_b[int(s[0])] = s[1]
			b_to_ib[s[1]] = int(s[0])
	return id_to_b, b_to_ib

def name_to_id(path, files,
			   ent_to_id,
			   rel_to_id):
	for file in files:
		strs_file = path+'/'+file+'.txt'
		id_file = path+'/'+file
		ids = []
		with open(strs_file,'r',encoding='utf-8') as f:
			for line in f:
				line = line.strip('\n')
				s = line.split('\t')
				#print(s)
				ids.append([str(ent_to_id[s[0]]),str(rel_to_id[s[1]]),str(ent_to_id[s[2]])])
		with open(id_file, 'w') as f:
			for id in ids:
				f.write('\t'.join(id)+'\n')
		
		

if __name__ == '__main__':
	
	path = '../data/FB237'
	files = ['test', 'valid', 'train']
	id_to_ent, ent_to_id, id_to_rel, rel_to_id = sync_id_and_name(path,files)
	# name_to_id('../data/Trans-test',['s1','s2','s3'],ent_to_id,rel_to_id)
	'''
	ent_to_id = {}
	id_to_ent = {}
	rel_to_id = {}
	id_to_rel = {}
	for file in ['train','test','valid']:
		ls = []
		with open('../data/Non-CommCom/'+file+'.txt','r',encoding='utf-8') as f:
			for line in f:
				line = line.strip('\n')
				s = line.split(' ')
				if s[0] not in ent_to_id:
					n = len(id_to_ent)
					id_to_ent[n] = s[0]
					ent_to_id[s[0]] = n
				
				if s[1] not in rel_to_id:
					n = len(id_to_rel)
					id_to_rel[n] = s[1]
					rel_to_id[s[1]] = n
				
				if s[2] not in ent_to_id:
					n = len(id_to_ent)
					id_to_ent[n] = s[2]
					ent_to_id[s[2]] = n
				ls.append([ent_to_id[s[0]], rel_to_id[s[1]], ent_to_id[s[2]]])
		with open('../data/Non-CommCom/'+file,'w') as f:
			for h,r,t in ls:
				f.write(str(h)+'\t'+str(r)+'\t'+str(t)+'\n')
	
	with open('../data/Non-CommCom/' + 'ent.dict', 'w', encoding='utf-8') as f:
		for k, v in id_to_ent.items():
			f.write(str(k) + '\t' + v + '\n')
	
	with open('../data/Non-CommCom/' + 'rel.dict', 'w', encoding='utf-8') as f:
		for k, v in id_to_rel.items():
			f.write(str(k) + '\t' + v + '\n')
	'''
		

	
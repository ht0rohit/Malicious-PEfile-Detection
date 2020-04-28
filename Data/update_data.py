""" This file extracts the required information of a given file selecting only those
attributes important from the viewpoint of a malware researcher,using the PE library """

import pefile
import os
import array
import math
import csv, json


def get_entropy(data):
	""" learn Entropy calculated across various sections & resources """
	if len(data) == 0:
		return 0.0
		
	occurences = array.array('L', [0]*256)
	for x in data:
		occurences[x if isinstance(x, int) else ord(x)] += 1

	entropy = 0
	for x in occurences:
		if x:
			p_x = float(x) / len(data)
			entropy -= p_x*math.log(p_x, 2)

	return entropy

	
def get_resources(pe):
	""" Extract resources : [entropy, size] """
	resources = []
	
	if hasattr(pe, 'DIRECTORY_ENTRY_RESOURCE'):
		try:
			for resource_type in pe.DIRECTORY_ENTRY_RESOURCE.entries:
				if hasattr(resource_type, 'directory'):
					for resource_id in resource_type.directory.entries:
						if hasattr(resource_id, 'directory'):
							for resource_lang in resource_id.directory.entries:
								data = pe.get_data(resource_lang.data.struct.OffsetToData, resource_lang.data.struct.Size)
								size = resource_lang.data.struct.Size
								entropy = get_entropy(data)

								resources.append([entropy, size])
		except Exception as e:
			return resources
			
	return resources

	
def get_version_info(pe):
	""" Return version info """
	res = {}
	
	for fileinfo in pe.FileInfo:
		if fileinfo.Key == 'StringFileInfo':
			for st in fileinfo.StringTable:
				for entry in st.entries.items():
					res[entry[0]] = entry[1]
		if fileinfo.Key == 'VarFileInfo':
			for var in fileinfo.Var:
				res[var.entry.items()[0][0]] = var.entry.items()[0][1]
				
	if hasattr(pe, 'VS_FIXEDFILEINFO'):
		  res['flags'] = pe.VS_FIXEDFILEINFO.FileFlags
		  res['os'] = pe.VS_FIXEDFILEINFO.FileOS
		  res['type'] = pe.VS_FIXEDFILEINFO.FileType
		  res['file_version'] = pe.VS_FIXEDFILEINFO.FileVersionLS
		  res['product_version'] = pe.VS_FIXEDFILEINFO.ProductVersionLS
		  res['signature'] = pe.VS_FIXEDFILEINFO.Signature
		  res['struct_version'] = pe.VS_FIXEDFILEINFO.StrucVersion
	
	return res


def extract_infos(fpath):
	""" Extract the attributes for a given PEfile """
	res = {}
	
	try:
		#check if file is a PEfile or not
		pe = pefile.PE(fpath)
	except:
		return -1
		
	res['Machine'] = pe.FILE_HEADER.Machine
	res['SizeOfOptionalHeader'] = pe.FILE_HEADER.SizeOfOptionalHeader
	res['Characteristics'] = pe.FILE_HEADER.Characteristics
	res['MajorLinkerVersion'] = pe.OPTIONAL_HEADER.MajorLinkerVersion
	res['MinorLinkerVersion'] = pe.OPTIONAL_HEADER.MinorLinkerVersion
	res['SizeOfCode'] = pe.OPTIONAL_HEADER.SizeOfCode
	res['SizeOfInitializedData'] = pe.OPTIONAL_HEADER.SizeOfInitializedData
	res['SizeOfUninitializedData'] = pe.OPTIONAL_HEADER.SizeOfUninitializedData
	res['AddressOfEntryPoint'] = pe.OPTIONAL_HEADER.AddressOfEntryPoint
	res['BaseOfCode'] = pe.OPTIONAL_HEADER.BaseOfCode
	try:
		res['BaseOfData'] = pe.OPTIONAL_HEADER.BaseOfData
	except AttributeError:
		res['BaseOfData'] = 0
	res['ImageBase'] = pe.OPTIONAL_HEADER.ImageBase
	res['SectionAlignment'] = pe.OPTIONAL_HEADER.SectionAlignment
	res['FileAlignment'] = pe.OPTIONAL_HEADER.FileAlignment
	res['MajorOperatingSystemVersion'] = pe.OPTIONAL_HEADER.MajorOperatingSystemVersion
	res['MinorOperatingSystemVersion'] = pe.OPTIONAL_HEADER.MinorOperatingSystemVersion
	res['MajorImageVersion'] = pe.OPTIONAL_HEADER.MajorImageVersion
	res['MinorImageVersion'] = pe.OPTIONAL_HEADER.MinorImageVersion
	res['MajorSubsystemVersion'] = pe.OPTIONAL_HEADER.MajorSubsystemVersion
	res['MinorSubsystemVersion'] = pe.OPTIONAL_HEADER.MinorSubsystemVersion
	res['SizeOfImage'] = pe.OPTIONAL_HEADER.SizeOfImage
	res['SizeOfHeaders'] = pe.OPTIONAL_HEADER.SizeOfHeaders
	res['CheckSum'] = pe.OPTIONAL_HEADER.CheckSum
	res['Subsystem'] = pe.OPTIONAL_HEADER.Subsystem
	res['DllCharacteristics'] = pe.OPTIONAL_HEADER.DllCharacteristics
	res['SizeOfStackReserve'] = pe.OPTIONAL_HEADER.SizeOfStackReserve
	res['SizeOfStackCommit'] = pe.OPTIONAL_HEADER.SizeOfStackCommit
	res['SizeOfHeapReserve'] = pe.OPTIONAL_HEADER.SizeOfHeapReserve
	res['SizeOfHeapCommit'] = pe.OPTIONAL_HEADER.SizeOfHeapCommit
	res['LoaderFlags'] = pe.OPTIONAL_HEADER.LoaderFlags
	res['NumberOfRvaAndSizes'] = pe.OPTIONAL_HEADER.NumberOfRvaAndSizes

	# Sections
	res['SectionsNb'] = len(pe.sections)
	entropy = list(map(lambda x:x.get_entropy(), pe.sections))
	res['SectionsMeanEntropy'] = sum(entropy)/float(len(entropy))
	res['SectionsMinEntropy'] = min(entropy)
	res['SectionsMaxEntropy'] = max(entropy)
	raw_sizes = list(map(lambda x:x.SizeOfRawData, pe.sections))
	res['SectionsMeanRawsize'] = sum(raw_sizes)/float(len(raw_sizes))
	res['SectionsMinRawsize'] = min(raw_sizes)
	res['SectionsMaxRawsize'] = max(raw_sizes)
	virtual_sizes = list(map(lambda x:x.Misc_VirtualSize, pe.sections))
	res['SectionsMeanVirtualsize'] = sum(virtual_sizes)/float(len(virtual_sizes))
	res['SectionsMinVirtualsize'] = min(virtual_sizes)
	res['SectionMaxVirtualsize'] = max(virtual_sizes)

	#Imports
	try:
		res['ImportsNbDLL'] = len(pe.DIRECTORY_ENTRY_IMPORT)
		imports = sum([x.imports for x in pe.DIRECTORY_ENTRY_IMPORT], [])
		res['ImportsNb'] = len(imports)
		res['ImportsNbOrdinal'] = len(list(filter(lambda x:x.name is None, imports)))
	except AttributeError:
		# No import
		res['ImportsNbDLL'] = 0
		res['ImportsNb'] = 0
		res['ImportsNbOrdinal'] = 0

	#Exports
	try:
		res['ExportNb'] = len(pe.DIRECTORY_ENTRY_EXPORT.symbols)
	except AttributeError:
		# No export
		res['ExportNb'] = 0
	
	#Resources
	resources= get_resources(pe)
	res['ResourcesNb'] = len(resources)
	
	if len(resources)> 0:
		entropy = list(map(lambda x:x[0], resources))
		res['ResourcesMeanEntropy'] = sum(entropy)/float(len(entropy))
		res['ResourcesMinEntropy'] = min(entropy)
		res['ResourcesMaxEntropy'] = max(entropy)
		sizes = list(map(lambda x:x[1], resources))
		res['ResourcesMeanSize'] = sum(sizes)/float(len(sizes))
		res['ResourcesMinSize'] = min(sizes)
		res['ResourcesMaxSize'] = max(sizes)
	else:
		res['ResourcesNb'] = 0
		res['ResourcesMeanEntropy'] = 0
		res['ResourcesMinEntropy'] = 0
		res['ResourcesMaxEntropy'] = 0
		res['ResourcesMeanSize'] = 0
		res['ResourcesMinSize'] = 0
		res['ResourcesMaxSize'] = 0

	# Load configuration size
	try:
		res['LoadConfigurationSize'] = pe.DIRECTORY_ENTRY_LOAD_CONFIG.struct.Size
	except AttributeError:
		res['LoadConfigurationSize'] = 0

	# Version configuration size
	try:
		version_infos = get_version_info(pe)
		res['VersionInformationSize'] = len(version_infos.keys())
	except AttributeError:
		res['VersionInformationSize'] = 0
	
	return res


if __name__ == '__main__':
	""" Extract information from binaries of benign PEfiles collected from 
	various sources & add it to the existing database """
	
	lis = []
	os.chdir(r"E:\00Malicious-PEfile-Detection")
	lis = os.listdir('Benign train/')
	
	#store infos as a collection of json lines
	str = '{"Details":['
	with open("Data/data.jsonl", "a") as outfile:
		for i in range(len(lis)):
			path = 'Benign train/' + lis[i]
			data = extract_infos(path)
		
			if(data!=-1):
				json_object = json.dumps(data)
				
				#write to data.jsonl 
				if (i == 0):
					outfile.write('{"Details":[' + json_object + ',')
				elif (i != len(lis) - 1):
					outfile.write(json_object + ',')
				else:
					outfile.write(json_object + ']}')
				#print(i)
	
	#update data.csv (filename changed to updated_data)
	data_file = open('Data/updated_data.csv', 'a')
	csv_writer = csv.writer(data_file)
	 
	with open('Data/data.jsonl') as json_file: #handles closing of json_file as well
		data = json.load(json_file)
		element = data['Details']
		count = 0
		
		for elem in element: 
			if count == 0:
				header = elem.keys() 
				csv_writer.writerow(header) 
				count += 1
	 
			#update some things manually, afterwards
			#name and md5 sections set to -1 (dropping it anyway)
			csv_writer.writerow(elem.values())
		
	data_file.close()
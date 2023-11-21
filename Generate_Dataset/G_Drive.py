from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive

gauth = GoogleAuth()
gauth.LocalWebserverAuth()
# gauth.CommandLineAuth()

drive = GoogleDrive(gauth)
IMG_PATH = 'F:/Images/STI/'
PHASE_PATH = IMG_PATH + 'Phase/'
CHI_PATH = IMG_PATH + 'Chi/'

file_list = drive.ListFile({'q': "'root' in parents and trashed=false"}).GetList()


def find_id(target, files):
    index = 0
    for file in files:
        actual_file = file['title']
        if actual_file == target:
            print('title: {}, id: {}'.format(file['title'], file['id']))
            return file['id']
        index += 1
        if index == len(files):
            print('File not found')
            return None


def list_files(file_id, gdrive):
    return gdrive.ListFile({'q': "'%s' in parents and trashed=false" % file_id}).GetList()


def get_id(path, gdrive):
    actual_id = 'root'
    for folder in path:
        folder_list = list_files(actual_id, gdrive)
        actual_id = find_id(folder, folder_list)
    print('')
    return actual_id


path_phase = ['Investigacion', 'Tesis', 'Images', 'Phase']
path_chi = ['Investigacion', 'Tesis', 'Images', 'Chi']


phase_id = get_id(path_phase, drive)
chi_id = get_id(path_chi, drive)

print('Listing phase files')
phase_list = list_files(phase_id, drive)
print('Listing chi files')
chi_list = list_files(chi_id, drive)

print('Done \n')
print('Saving images files')
for n in range(1478, 1990):
    phase_file = drive.CreateFile({'id': phase_list[n]['id']})
    phase_name = PHASE_PATH + phase_file['title']
    phase_file.GetContentFile(phase_name)
    print(phase_name + ' saved')

    chi_file = drive.CreateFile({'id': chi_list[n]['id']})
    chi_name = CHI_PATH + chi_file['title']
    chi_file.GetContentFile(chi_name)
    print(chi_name + ' saved')

print('Done')

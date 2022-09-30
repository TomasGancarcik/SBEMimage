from os.path import splitext

fn = '20220831_Ruth_20220426_RM0008_130hpf_fP1_f3_run001_g0001_t0587_s100000.tif'

def get_predecessor_filename(fn):
    base, ext = splitext(fn)
    base_excl_slice_nr, slice_nr_str = str.split(base, '_s')
    zeros_count = 0
    
    if int(slice_nr_str) == 1:
        fn_predecessor = None
    else:
        while slice_nr_str[0] == '0':
            slice_nr_str = slice_nr_str[1:]
            zeros_count +=1
        fn_predecessor = base_excl_slice_nr + '_s' + '0'*zeros_count + str(int(slice_nr_str)-1) + ext
    return fn_predecessor
    
new = get_predecessor_filename(fn)

print('reference name: ', fn)
print('new name: ', new)
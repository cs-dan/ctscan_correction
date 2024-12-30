### Unit Tests

# Testing: get_run method
for i in range(5):
    run_folder = get_run()

# Testing: write_log method
run_info = ['file1', 200, 10, 0.34334331]
run_folder = get_run()
log_file = write_log(run_folder, run_info)

# Testing: save_frame method
object = np.random.rand(256,256)
names = [] 
images = []
path = f'results\\run_1'

fig = pyp.figure(figsize= (12, 4))
plt.imshow(object, cmap= plt.cm.Greys_r, vmin= 0, vmax= 1)
pyp.title(f"Image")

images.append(fig)
names.append('object1')

save_to_disk(path, images, names)

# Testing: single line multi return
def eff():
    return 3, 5

list1 = []
list2 = []
_ = (lambda x: list1.append(x[0]) or list2.append(x[1]))(eff())
print(list1); print(list2)

# Testing: splitting on filename
my_string = "example_string_with_several_parts"
result = '_'.join(my_string.split('_')[3:4])
print(result)
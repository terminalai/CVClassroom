import keras 

import aiclass 
import dataloader 
import utils 
import params 

import os 


keras.utils.set_random_seed(params.seed) 


# variables 
n = 3 
initializer = aiclass.Model_Conv2D_Dense 
init_kwargs = {
    'conv2d_input_shape': (32, 32, 3), # CIFAR-10 standard 
    'conv2d_kwargss': list, 
    'dense_input_shape': tuple, 
    'dense_kwargss': list
} 
default_train_kwargs = {
    
} 


# MAIN LOOP STUFF 

def load_model_from_dir(dir, technique="init"): 
    # find out which saved model to use 
    fin = open(dir+technique+".txt", 'r') 
    path = fin.readline().strip() 
    fin.close() 

    # load model from file 
    model = initializer(**init_kwargs, save_checkpoint_dir=dir) 
    model.load(path) 
    return model 


def init(train_kwargs=default_train_kwargs): 
    parent_dir = "./classrooom_models/iter0/" 

    try: 
        os.mkdir(parent_dir) 
    except: 
        pass 

    for i in range(n*n): # create n^2 students
        save_dir = parent_dir+"student"+str(i)+"/" 
        student = initializer(**init_kwargs, save_checkpoint_dir=save_dir) 
        student.save(save_dir+"init.ckpt")

        fout = open(save_dir+"last.txt", 'r') 
        fout.write(save_dir+"init.ckpt\n") # save that as the last so far 
        fout.close() 

        fout = open(save_dir+"best.txt", 'r') 
        fout.write(save_dir+"init.ckpt\n") # save that as the best so far 
        fout.close() 
    del student # delete student object to free storage space

    for i in range(n): 
        save_dir = "./classroom_models/iter0/teacher"+str(i)+"/"
        # create teacher and save at "./classroom_models/iter0teacher"+str(i) 
        teacher = initializer(**init_kwargs, save_checkpoint_dir=save_dir) 
        # train teacher 
        results = teacher.train(dataloader.x_train, dataloader.y_train, **train_kwargs) 

        # save best results teacher location 
        fout = open(save_dir+"best.txt", 'r') 
        fout.write() # TODO 
        fout.close() 

        # save last epoch teacher location 
        try: 
            last_epoch = train_kwargs['epochs'] 
        except: 
            last_epoch = initializer.default_epoch_count 
        fout = open(save_dir+"last.txt", 'r') 
        fout.write(save_dir+"epoch_{epoch:04d}.ckpt".format(epoch=last_epoch)) 
        fout.close() 

    del teacher # delete teacher object to free storage space


# function to teach each of n^2 students using for loop
def teach_students(pseudolabels, iterno, train_kwargs=default_train_kwargs, studentidxs=range(n*n), return_student_results=False): # maybe have default teacher be first iteration teacher model thing ig (i think it was just all the right data) 
    parent_dir = "./classrooom_models/iter"+str(iterno)+"/" 
    if return_student_results: student_results = [] 
    for idx in studentidxs: 
        save_dir = parent_dir+"student"+str(idx)+"/" 

        student = load_model_from_dir(save_dir, technique="last") 

        # train student 
        results = student.train(dataloader.x_train, pseudolabels, **train_kwargs)

        if return_student_results: student_results.append(results) 

        # save last epoch student location 
        try: 
            last_epoch = train_kwargs['epochs'] 
        except: 
            last_epoch = initializer.default_epoch_count 
        fout = open(save_dir+"last.txt", 'r') 
        fout.write(save_dir+"epoch_{epoch:04d}.ckpt".format(epoch=last_epoch)) 
        fout.close() 
    del student 

    if return_student_results: return student_results 


def iterate(iterno, technique="best", show_msgs=False): 
    parent_dir = "./classrooom_models/iter"+str(iterno)+"/" 
    for tidx in range(n): 
        teacher_dir = parent_dir+"teacher"+str(tidx)+"/" 
        teacher = load_model_from_dir(teacher_dir, technique=technique) 

        # make data using pseudolabels from teachers 
        pseudolabels = teacher.sample(dataloader.x_train) 

        del teacher # save memory 

        if show_msgs: print("TEACHER",tidx,"IS TEACHING")
        if (tidx != n): 
            teach_students(pseudolabels, iterno) # each teacher teaches all n^2 students
        else: 
            student_results = teach_students(pseudolabels, iterno, return_student_results=True) 
        if show_msgs: print("FINISHED TEACHING") 
    
    # all students have been taught 

    # TODO: FIGURE OUT HOW OT USE STUDENT RESULTS TO PROMOTE STUDENTS TO TEACHERS 
    target_student_dirs = [] 

    try: 
        os.mkdir("./classrooom_models/iter"+str(iterno+1)+"/") 
    except: 
        pass 

    for i in range(n): 
        # promote student to teacher 

        new_teacher_dir = "./classroom_models/iter"+str(iterno+1)+"/teacher"+str(i)+"/" 
        newteacher = initializer(**init_kwargs, save_checkpoint_dir=new_teacher_dir) 
        newteacher.load(target_student_dirs[i]) 
        newteacher.save(new_teacher_dir+"init.ckpt")

        fout = open(new_teacher_dir+"last.txt", 'r') 
        fout.write(new_teacher_dir+"init.ckpt\n") # save that as the last so far 
        fout.close() 

        fout = open(new_teacher_dir+"best.txt", 'r') 
        fout.write(new_teacher_dir+"init.ckpt\n") # save that as the best so far 
        fout.close() 
        del newteacher 

    # TODO: decide whether we have partially trained students who failed train again, or just make completely new students 
    for i in range(n*n): 
        new_student_dir = "./classroom_models/iter"+str(iterno+1)+"/student"+str(i)+"/" 
        newstudent = initializer(**init_kwargs, save_checkpoint_dir=new_student_dir)
        newstudent.save(new_student_dir+"init.ckpt") 

        fout = open(new_student_dir+"last.txt", 'r') 
        fout.write(new_student_dir+"init.ckpt\n") # save that as the last so far 
        fout.close() 

        fout = open(new_student_dir+"best.txt", 'r') 
        fout.write(new_student_dir+"init.ckpt\n") # save that as the best so far 
        fout.close() 
        del newstudent 

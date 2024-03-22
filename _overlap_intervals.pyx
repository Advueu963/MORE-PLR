"""
    This file contains a Cython DFS implementation to find overlapping intervals.
"""

# cython: language_level=3
# cython: cdivision=True
# cython: boundscheck=False
# cython: wraparound=False
import numpy as np
cimport numpy as np
from _overlap_intervals cimport DTYPE_t,DTYPE_t_1D,DTYPE_t_2D
from libc.stdio cimport printf
from libc.stdlib cimport malloc,free
from libc.math cimport lround as c_lround
#from cpython.mem cimport PyMem_Malloc, PyMem_Realloc, PyMem_Free


# utils functions to create and free weighted adjacency matrix
cdef DTYPE_t **get_overlaps_array(np.int64_t n_classes,np.int64_t values_in_classes, DTYPE_t init_value) nogil:
    cdef DTYPE_t  **pointer
    pointer = <DTYPE_t **> malloc(sizeof(DTYPE_t *) * n_classes)
    if pointer == NULL:
        printf("ERROR with allocating memory in get_overlaps_array")
    for k in range(values_in_classes):
       pointer[k] = <DTYPE_t*> malloc(sizeof(DTYPE_t) * values_in_classes)
       if not pointer[k]:
            for j in range(k):
               free(pointer[k])
            free(pointer)
            printf("ERROR with allocating memory in get_overlaps_array\n")
            return NULL

    for i in range(n_classes):
        for j in range(values_in_classes):
            pointer[i][j] = init_value
    return pointer

cdef void free_overlaps_array(DTYPE_t **pointer, np.int64_t n_classes) nogil:
    #printf("LOW LEVEL FREE\n")
    for k in range(n_classes):
        free(pointer[k])
    #printf("TOP LEVEL FREE\n")
    #printf("OF THIS POINTER %p\n", <void *> pointer)
    free(pointer)

cdef void print_overlaps_array(DTYPE_t **overlaps_interval, np.int64_t n_classes, np.int64_t values_in_classes) nogil:
    for i in range(n_classes):
        for j in range(values_in_classes):
            printf("%.3f ", overlaps_interval[i][j])
        printf("\n")

cdef void dfs_min_val(DTYPE_t** overlaps, np.int64_t n_classes,
                      np.uint8_t* visited, np.int64_t start, DTYPE_t *min_val,
                      DTYPE_t* group, np.int64_t* free_pos) nogil:
    ##printf("ENTRED min val dfs %d\n", start)
    visited[start] = 1
    cdef DTYPE_t pot_new_min # potential new min of bucket with start
    for i in range(n_classes):
        pot_new_min = overlaps[start][i]
        if visited[i] == 0 and pot_new_min > 0:
            # add ith class to the overlap group of start
            addElementToGroup(group,free_pos,i)

            # Check if ith class is newest minimial boundary of interval group
            if min_val[0] > pot_new_min:
                min_val[0] = pot_new_min
            dfs_min_val(overlaps, n_classes, visited, i, min_val, group, free_pos)

cdef void addElementToGroup(DTYPE_t *group, np.int64_t *free_pos, DTYPE_t element) nogil :
    group[free_pos[0]] = element
    free_pos[0] = free_pos[0] + 1

cdef DTYPE_t** dfs(DTYPE_t** overlaps,np.int64_t n_classes, np.uint8_t* visited, np.int64_t start) nogil:
    # Start DFS Search for connected components
    cdef DTYPE_t min_value

    # We allocate n_classes +1 for the first dimension because only allocating n_classes yields in an error (SIGABRT).-
    # While this does create a certain memory overhead, the main algorithm is not influenced by this.
    cdef DTYPE_t** merged_overlaps = get_overlaps_array(n_classes+1,n_classes+1,-1) # n_classes x (n_classes + 1) matrix

    #print_overlaps_array(merged_overlaps, n_classes,n_classes)

    cdef DTYPE_t *group
    cdef np.int64_t* pos = <np.int64_t*> malloc(sizeof(np.int64_t)*1) # next position in merged_overlaps[i] were the index of an overlap interval can be placed
    for group_leader in range(n_classes):
        pos[0] = 1
        group = merged_overlaps[group_leader]
        if visited[group_leader] == 0:
            visited[group_leader] = 1
            min_value = overlaps[group_leader][group_leader]
            addElementToGroup(group, pos, group_leader)
            for i in range(n_classes):
                if visited[i] == 0 and overlaps[group_leader][i] > 0:
                    addElementToGroup(group,pos,i)
                    ##printf("%d entred under %d\n",i, group_leader)
                    if min_value > overlaps[group_leader][i]:
                        min_value = overlaps[group_leader][i]
                    ##printf("min border of %d overlapping with %d is %.3f",start,i, overlaps[start][i])
                    dfs_min_val(overlaps,n_classes,visited,i,&min_value,group,pos)
                    ##printf("ENDED GROUP SEARCH\n")
                # now we gathered all connected Components of i
            pos[0] = 0
            addElementToGroup(group,pos,min_value)


   # #printf("END:\n")
   # #printf("Pointer Adresses \n")
   # #printf("%p\n", <void *> merged_overlaps)
   # for i in range(n_classes):
   #     #printf("%p\n", <void *> merged_overlaps[i])
   # #printf("Pointer Adress END\n")

    #printf("GROUP POINTER:\n")
    #printf("%p\n", <void *> group)


    #printf("POS POINTER:\n")
    #printf("%p\n", <void*> pos)
    #free_overlaps_array(merged_overlaps,n_classes+1)
    free(pos)
    return merged_overlaps


cdef void build_ranking(DTYPE_t **non_overlap_groups,
                        np.int64_t n_classes,
                        DTYPE_t_1D consensus) nogil :
    # get the index order to have a ascending order of the unique buckets
    cdef DTYPE_t min_val_group
    cdef DTYPE_t min_val_otherGroup
    cdef np.int64_t how_often_bigger
    cdef np.int64_t unique_buckets = 0  # save amount of unique buckets to add elements to correct position in ranking
    cdef np.int64_t *descending_rank_position = <np.int64_t *> malloc(sizeof(np.int64_t) * n_classes)

    if descending_rank_position is NULL:
        #printf("Error: Memory allocation failed for ascending_rank_position\n")
        return  # or handle the error in an appropriate way
    #printf("ENDED ASCENDING RANK POSITION PyMem_Malloc\n")
    #printf("STARTED ASCENDING RANK POSITION\n")
    for group in range(n_classes):
        min_val_group = non_overlap_groups[group][0]
        if min_val_group < 0:
            descending_rank_position[group] = -1
        else:
            unique_buckets = unique_buckets + 1
            how_often_bigger = 0
            for other_group in range(n_classes):
                min_val_otherGroup = non_overlap_groups[other_group][0]
                if min_val_group > min_val_otherGroup and min_val_otherGroup >= 0:
                    # min_val_group is ranked before min_val_otherGroup
                    how_often_bigger = how_often_bigger + 1
            descending_rank_position[group] = how_often_bigger

    cdef np.int64_t bucket_rank
    cdef np.int64_t elem
    cdef np.int64_t interval_num

    # insert in y the rankings of the classes
    for bucket in range(n_classes):
        # iterate over each ascending_rank_position
        bucket_rank = descending_rank_position[bucket] + 1  # this + 1 is because of sklearn interface
        if bucket_rank > 0:  # Check if this bucket even truly exists --> bucket_rank >= 1 --> ascending_rank_position >= 0
            for i in range(1, n_classes + 1):
                # Each Element in this bucket gets its corresponding value
                elem = <np.int64_t> non_overlap_groups[bucket][i]
                interval_num =  c_lround(non_overlap_groups[bucket][0])
                #printf("interval_num: %d vs original: %f\n",interval_num, non_overlap_groups[bucket][0])
                if elem >= 0:
                    consensus[elem] = bucket_rank # !!! actually filling the prediction vector !!!
    #printf("STARTING FREE of ranking build\n")
    free(descending_rank_position)
    #printf("ASCENDING FREE DONE\n")
    #free_overlaps_array(non_overlap_groups,n_classes)
    #printf("ENDED FREE of ranking build\n")

cdef void _get_overlaps(DTYPE_t_2D intervals,
                        np.int64_t n_classes,
                        DTYPE_t_1D consensus) nogil:
    # Allocate memory for  adjacency matrix called overlaps
    cdef DTYPE_t **overlaps = get_overlaps_array(n_classes,n_classes,0)
    if overlaps is NULL:
        return

    # bunch of variable declarations
    cdef DTYPE_t **non_overlap_groups
    cdef np.uint8_t *visited = <np.uint8_t*> malloc(sizeof(np.uint8_t)*n_classes)
    cdef float current_class_lower
    cdef float current_class_upper
    cdef float other_class_lower
    cdef float other_class_upper
    for n in range(n_classes):
        visited[n] = 0

    # Fill the matrix
    for i in range(n_classes):
        current_class_lower = intervals[i][0]
        current_class_upper = intervals[i][1]
        # now check for each other interval if it overlaps with current_class
        for j in range(n_classes):
            # iterate over each other class
            other_class_lower = intervals[j][0]
            other_class_upper = intervals[j][1]
            #https://stackoverflow.com/questions/3269434/whats-the-most-efficient-way-to-test-if-two-ranges-overlap
            if current_class_lower <= other_class_upper and other_class_lower <= current_class_upper:
                overlaps[i][j] = min(current_class_lower,other_class_lower) # the value indicating that i and j overlaps is the minimal value of the "new" bigger interval

    # find all overlapping intervals and distinguish them in groups
    # the groups are the rows and the columns the indices of the intervals belonging to the group
    # -1 indicates not a group / not a member
    non_overlap_groups = dfs(overlaps,n_classes,visited,0)

    # Get the mean prediction of the groups interval as new representative of the groups
    cdef int group_member
    cdef float current_group_lower
    cdef float current_group_upper
    for group_number in range(n_classes): # last group is memory overhead
        # Initialize the group lower and upper values
        group_member = <np.int8_t> non_overlap_groups[group_number][1]
        if group_member >= 0: # this is a group with at least 1 valid member
            current_group_lower = intervals[group_member][0]
            current_group_upper = intervals[group_member][1]
            for possible_valid_members in range(2,n_classes+1): # lets check for other valid group member
                group_member = <np.int8_t> non_overlap_groups[group_number][possible_valid_members]
                if group_member >=0: # found another valid group member
                    # update the lower and upper bound of the group
                    current_group_lower = min(current_group_lower, intervals[group_member][0])
                    current_group_upper = max(current_group_upper, intervals[group_member][0])

            # update the representing value of the group
            non_overlap_groups[group_number][0] = (current_group_lower + current_group_upper) / 2

    free(visited)
    #printf("ENDED DFS\n")

    # Now thake these new groups in bring them in the api dense ranking position vector format
    build_ranking(non_overlap_groups=non_overlap_groups,n_classes=n_classes,consensus=consensus)

    # clean up
    free_overlaps_array(overlaps,n_classes)
    free_overlaps_array(non_overlap_groups,n_classes+1)


cpdef test(intervals,n_classes,consensus):
    _get_overlaps(intervals,n_classes, consensus)

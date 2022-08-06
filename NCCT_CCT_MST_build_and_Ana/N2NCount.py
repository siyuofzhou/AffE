from Ana.DataLoader import All_the_Data_You_Need, get_rev_rel, BuildDataDict


def countN2N(dataset = 'WN18RR'):
    all_data = All_the_Data_You_Need(dataset=dataset)
    n2n = 0
    for h,r,t in all_data.data["all_trips"]:
        r_rev = get_rev_rel(r, all_data.rel_num)
        if len(all_data.hr_t[(h, r)]) > 5 and len(all_data.hr_t[(t, r_rev)]) > 5:
            n2n += 1
    print(n2n/len(all_data.data["all_trips"]))


if __name__ == "__main__":
    BuildDataDict().build("YAGO3-10")
    countN2N("WN18RR")
    countN2N("FB237")
    countN2N("YAGO3-10")



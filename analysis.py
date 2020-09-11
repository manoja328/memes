
idx = []
for i, ent in enumerate(submission.to_dict('r')):
    p, label = ent['proba'], ent['label']
    if label == 0 and p > 0.4:
        idx.append(i)
    elif label == 1 and p < 0.4:
        idx.append(i)

# In[64]:


# j1 = test_dataset.samples_frame.iloc[idx,:]
# j2 = submission.iloc[idx,:]
# ct = pd.concat([j1,j2],axis=1).reset_index()
# ct.head()

ct = pd.concat([test_dataset.samples_frame, submission], axis=1).reset_index()
ct.head()

# In[65]:


from PIL import Image
from IPython.core.display import HTML

images = []
for i in np.random.randint(0, len(ct), size=4):
    print(ct.loc[i, "label"], ct.loc[i, "text"])
    #     st = '<img src = {}></img>'.format(ct.loc[i, "img"])
    #     st = '<img src="{}">click here</img>'.format("http://129.21.57.119:8889/tree/data/img/97015.png")
    #     HTML(st)
    images.append(
        Image.open(ct.loc[i, "img"]).convert("RGB")
    )

# convert the images and prepare for visualization.
tensor_img = torch.stack(
    [simple_transform(image) for image in images]
)
grid = torchvision.utils.make_grid(tensor_img, nrow=2)

# pl
plt.figure(figsize=(20, 10))
plt.axis('off')
plt.imshow(grid.permute(1, 2, 0))

# # how can we use complemetray images as in VQA 2

# # read the paper and see how words are associated with 0 and 1 ??

# In[160]:


# train_samples_frame[train_samples_frame.text.str.contains("pussy")]
ct[ct.text.str.contains("cat")]

# In[164]:


row = ct.iloc[739]
img = Image.open(row.img).convert("RGB")
plt.imshow(img)
plt.xlabel(row.text)

# In[107]:


aa = train_samples_frame.groupby("label")

# In[113]:


go = train_samples_frame.iloc[aa.groups[0]]

# In[117]:


cc = go.text.map(
    lambda text: text.split(" ")
).to_list()

# In[119]:


from collections import defaultdict

cnt = defaultdict(int)
for l in cc:
    for word in l:
        cnt[word] += 1

# In[124]:


# Most frequent items
{k: v for k, v in sorted(cnt.items(), key=lambda item: -item[1])}

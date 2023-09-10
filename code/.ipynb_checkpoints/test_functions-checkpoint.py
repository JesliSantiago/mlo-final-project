{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 1,
   "id": "5df16bb1",
   "metadata": {},
   "outputs": [],
   "source": [
    "# assert expected values\n",
    "# note for pytest to work, function must start with test_\n",
    "def test_data(df):\n",
    "\n",
    "    assert df['Poem'].apply(isinstance, args=(str,)).all(), \"Not all poems are strings.\"\n",
    "\n",
    "    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff']\n",
    "    assert not df['Poem'].str.endswith(tuple(image_extensions)).any(), \"Found potential image files in the Poem column.\"\n",
    "\n",
    "    assert not df.isnull().any().any(), \"There are missing values in the dataframe.\""
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "7f0e06ce",
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.9.13"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}



def _get_string_binary(number, lenght):
    """
    Generate a string of length 'length' of 'number' in binary.
        :param number: number in decimal
        :param lenght: lenght of the string
    """
    number_bin = format(number, "b")
    string_binary = '0'*lenght
    string_binary = string_binary[:lenght-len(number_bin)] + number_bin
    return string_binary


def _get_multilinear_value(channel, position):
    """
    Calculate the interpolated value for a given position in the
    channel content.
        channel: A N-D Tensor [D0, D1, ..., DN-1]
        position: A 1-D float32 Tensor of N elements. Contents the position in
                    the channel. 
    """
    N = position.shape[0]
    # Dictionary with the values around the target position and the fraction
    # of the position
    surrounds = {}
    
    # for each dimension extract the coordinates near to the position 'T'
    for d in range(N):
        integer_part = int(position[d])
        fraction_part = position[d] - integer_part
        integer_part_plus_one = min(integer_part + 1, channel.shape[d] - 1)
        
        surrounds[d] = {'0': integer_part,
                        '1': integer_part_plus_one,
                        'T': fraction_part}

    value = 0
    # for each element in the multilinear calculation
    for i in range(2**N):
        temp = 1.0
        # position of the element in the format '0000101'
        position_to_read = _get_string_binary(i, N)
        
        coordinate = []
        # l_r left or right, ic can be '0' or '1'
        for d, l_r in enumerate(reversed(range(len(position_to_read)))):
            # coordinate in the channel of the value multiplying the element 
            coordinate.append(surrounds.get(d).get(position_to_read[l_r]))
            
            # multiplication in the element
            if position_to_read[l_r] == '0':
                temp = temp * (1.0 - surrounds.get(d).get('T'))
            else:
                temp = temp * surrounds.get(d).get('T')
        
        # the element times the value at the coordinate in the channel
        temp = temp * channel.item(tuple(coordinate))
        # summing all the elements
        value = value + temp

    return int(value + 0.5)


def _resize_channel_multilinear(channel, size):
    """
    Resizes content in N dimensions to size using multiliniear interpolation.
    args:
        images: An (N + 2)-D Tensor [D0, D1, ..., DN-1]
        size: A 1-D int32 Tensor of N elements: new_0D, new_1D,..new_N-1D 
    """
    N = size.shape[0]
    # This is the only code limiting N, but can be extended easily
    assert N <= 4
    assert N >= 2
    
    # new image
    #resized_channel = tf.zeros(size, dtype=tf.int32)
    resized_channel = np.zeros(size, dtype = int)
    
    scale = []
    for d in range(N):
        scale.append(float(channel.shape[d]) / float(size[d]))
    # finds positions in the original image and interpolates
    for i in range(size[0]):
        for j in range(size[1]):
            if N > 2:
                for k in range(size[2]):
                    if N > 3:
                        for l in range(size[3]):
                            # Case for N = 4
                            position = np.ones((4), dtype=np.float32)
                            position[0] = i * scale[0]
                            position[1] = j * scale[1]
                            position[2] = k * scale[2]
                            position[3] = l * scale[3]
                            value = _get_multilinear_value(channel, position)
                            resized_channel[i, j, k, l] = value
                    else:
                        # Case for N = 3
                        position = np.ones((3), dtype=np.float32)
                        position[0] = i * scale[0]
                        position[1] = j * scale[1]
                        position[2] = k * scale[2]
                        value = _get_multilinear_value(channel, position)
                        resized_channel[i, j, k] = value
            else:
                # Case for N = 2
                position = np.ones((2), dtype=np.float32)
                position[0] = i * scale[0]
                position[1] = j * scale[1]
                value = _get_multilinear_value(channel, position)
                resized_channel[i, j] = value
    return resized_channel


def resize_multilinear(content, size):
    """
    Resizes content in N dimensions to size using multiliniear interpolation.
    args:
        images: An (N + 2)-D Tensor [batch, D0, D1, ..., DN-1, channels]
        size: A 1-D int32 Tensor of N elements: new_0D, new_1D,..new_N-1D 
    """
    batches = content.shape[0]
    channels = content.shape[-1]
    
    new_size = list(content.shape)
    new_size[1:len(new_size)-1] = size
    
    # new content
    # resized_channel = tf.zeros(size, dtype=tf.int32)   
    resized_content = np.zeros(new_size, dtype=int)
    
    for batch in range(batches):
        for channel in range(channels):
            img_channel = content[batch, ..., channel]
            resized_channel = _resize_channel_multilinear(img_channel, size)
            
            resized_content[batch,...,channel] = resized_channel

    return resized_content
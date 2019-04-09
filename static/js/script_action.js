// add update
$('body').on('click', '.modal-show', function (event) {
    event.preventDefault();
    var me = $(this),
        url = me.attr('href'),
        title = me.attr('title');
    $('#modal-title').text(title);
    // $('#modal-btn-save').text(me.hasClass('edit') ? 'Update' : 'Create');
    $('#modal-btn-save').removeClass('hide')
        .text(me.hasClass('edit') ? 'Update' : 'Create');

    $.ajax({
        url: url,
        dataType: 'html',
        success: function (response) {
            $('#modal-body').html(response);
        }
    });
    $('#modal').modal('show');
});

// show
$('body').on('click', '.btn-show', function (event) {
    event.preventDefault();
    var me = $(this),
        url = me.attr('href'),
        title = me.attr('title');
    $('#modal-title').text(title);
    $('#modal-btn-save').addClass('hide');
    $.ajax({
        url: url,
        dataType: 'html',
        success: function (response) {
            $('#modal-body').html(response);
        }
    });
    $('#modal').modal('show');
});

// delete
$('body').on('click', '.btn-delete', function (event) {
    event.preventDefault();
    var me = $(this),
        url = me.attr('href'),
        title = me.attr('title'),
        csrf_token = $('meta[name="csrf-token"]').attr('content');
    swal({
        title: 'Are you sure want to delete ' + title + ' ?',
        text: 'You won\'t be able to revert this!',
        type: 'warning',
        showCancelButton: true,
        confirmButtonColor: '#3085d6',
        cancelButtonColor: '#d33',
        confirmButtonText: 'Yes, delete it!'
    }).then((result) => {
        if (result.value) {
            $.ajax({
                url: url,
                type: "POST",
                data: {
                    '_method': 'DELETE',
                    '_token': csrf_token
                },
                success: function (response) {
                    $('#datatable').DataTable().ajax.reload();
                    swal({
                        type: 'success',
                        title: 'Success!',
                        text: 'Data has been deleted!',
                        timer: 1500

                    });
                },
                error: function (xhr) {
                    swal({
                        type: 'error',
                        title: 'Oops...',
                        text: 'Something went wrong!'
                    });
                }
            });
        }
    });
});

// action add update
$('#modal-btn-save').click(function (event) {
    event.preventDefault();
    // var form = $(this);
    // var formdata = false;
    // if (window.FormData){
    //     formdata = new FormData(form[0]);
    // }
    var form = $('#modal-body form'),
        url = form.attr('action')
    // method = $('input[name=_method]').val() == undefined ? 'POST' : 'POST';
    ;
    form.find('.help-block').remove();
    form.find('.form-group').removeClass('has-error');
    $.ajax({
        url: url,
        method: 'POST',
        // data : form.serialize(),
        data: new FormData($("#modal-body form")[0]),
        contentType: false,
        processData: false,
        beforeSend: function () {
            // $("body").css("padding", '0px');
            swal({
                title: 'Sedang Memuat...',
                onOpen: () => {
                    swal.showLoading()
                }
            }).catch(swal.noop);
        },
        success: function (response) {
            console.log(response);
            form.trigger('reset');
            $('#modal').modal('hide');
            $('#datatable').DataTable().ajax.reload();
            swal({
                type: 'success',
                title: 'Success!',
                text: 'Data has been saved!',
                timer: 1500
            });
        },
        error: function (xhr) {
            swal.close()
            var res = xhr.responseJSON;
            if ($.isEmptyObject(res) == false) {
                $.each(res.errors, function (key, value) {
                    $('#' + key)
                        .closest('.form-group')
                        .addClass('has-error')
                        .append('<span class="help-block"><strong>' + value + '</strong></span>');
                });
            }
        }
    })
});